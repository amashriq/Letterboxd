# CLI to predict a star rating for one movie, for the profile folded in from
# a --user's movielens_matched.csv (data/<user>-<date>/movielens_matched.csv).
# Looks a title up in MovieLens (title search REQUIRES --year -- see
# find_movie), scores it with the trained hybrid MF model. Two cases, from
# best to weakest signal:
#   warm -- has a trained item vector (>=MIN_RATINGS ratings): identity +
#           content embedding
#   cold -- everything else, treated identically regardless of whether the
#           movie has a MovieLens id at all: live TMDB fetch (via its
#           links.csv tmdbId if it has a MovieLens id, via a live title
#           search if it doesn't) + content embedding
# Prints the clamped (0.5-5.0) prediction for whichever case applies.
#
# Usage:
#   python scripts/predict.py --user adeeb --title "Mad Max: Fury Road" --year 2015
#   python scripts/predict.py --user adeeb --movie-id 122904
#   python scripts/predict.py --user caroline --title "Project Hail Mary" --year 2026

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.content import (
    load_hybrid_artifacts,
    load_content_store,
    get_content_row,
    get_item_representation,
    search_tmdb_by_title,
    fetch_movie_record,
)
from engine.match import load_tmdb_to_ml
from engine.paths import DATA_DIR, resolve_user_dir
from engine.recommend import fold_in_user, predict_rating, load_movie_metadata
from scripts.fetch_tmdb_content import GENRE_ID_TO_IDX, GENRE_NAMES

# Anchored to the end of the string: MovieLens titles are "Title (YYYY)",
# and a few have a duplicated/spurious extra "(YYYY)" group (e.g. "The
# Devotion of Suspect X (2017) (2017)") -- anchoring to $ still gets the
# right year in both that case and titles with an earlier non-year
# parenthetical, e.g. "...(Remaining Sense of Pain) (2008)".
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def find_movie(
    title_query: str, year: int, meta: dict[int, tuple[str, str]]
) -> list[tuple[int, str, str]]:
    """
    Case-insensitive substring search over (title, genres) metadata,
    filtered to titles whose own parsed year matches `year` exactly.

    Year is required (not optional) because a bare substring match alone is
    prone to large false-positive sets -- e.g. "Sinners" (the query) matches
    27 unrelated MovieLens titles containing that word, and title alone
    can't disambiguate genuine same-title-different-year cases like
    "Dune (1984)" / "Dune (2000)" / "Dune (2021)". A title with no
    parenthesized year at all (a real minority of MovieLens rows) can never
    match here -- use --movie-id for those instead.
    """
    q = title_query.lower()
    out = []
    for mid, (t, g) in meta.items():
        if q not in t.lower():
            continue
        m = YEAR_RE.search(t)
        if m and int(m.group(1)) == year:
            out.append((mid, t, g))
    return out


def get_tmdb_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        print("No TMDB_API_KEY set (add one to .env) -- required for any cold prediction.")
        sys.exit(1)
    return api_key


def print_prediction(
    title: str,
    genres: str,
    movie_id: int,
    status_label: str,
    raw_pred: float,
    clamped: float,
    already_rated: dict[int, float],
) -> None:
    print(f"{title}  [{genres}]  (movieId {movie_id})")
    print(f"  {status_label}")
    print(f"  raw predicted rating:     {raw_pred:.2f}")
    print(f"  clamped predicted rating: {clamped:.2f} / 5.0")
    if movie_id in already_rated:
        print(f"  (note: you already rated this {already_rated[movie_id]:.1f} -- this is an in-sample prediction)")


def resolve_via_live_tmdb(title_query: str, year: Optional[int]) -> tuple[int, str, str, dict]:
    """
    Fall back for a title with NO MovieLens id at all: search TMDB directly,
    fetch the full record, and return enough to build a content-only
    feat_row later. Raises SystemExit (via sys.exit) if nothing resolves.
    """
    api_key = get_tmdb_api_key()
    hit = asyncio.run(search_tmdb_by_title(title_query, api_key, year))
    if hit is None:
        print(f"{title_query!r} not found on TMDB either (no MovieLens entry, no TMDB match).")
        sys.exit(1)

    record = asyncio.run(fetch_movie_record(hit["id"], api_key))
    if record is None:
        print(f"Found {hit.get('title')!r} on TMDB (id {hit['id']}) but the detail fetch failed.")
        sys.exit(1)

    disp_title = hit.get("title") or title_query
    year_str = (hit.get("release_date") or "")[:4]
    if year_str:
        disp_title = f"{disp_title} ({year_str})"
    genre_names = [
        GENRE_NAMES[GENRE_ID_TO_IDX[gid]] for gid in record.get("genre_ids", []) if gid in GENRE_ID_TO_IDX
    ]
    genres = "|".join(genre_names) if genre_names else "(no genres listed)"

    # -1 is a safe sentinel: real MovieLens movieIds are always positive, so
    # it can never collide with a real row -- both lookups below cleanly
    # fall through to using `record` fresh instead of an existing row.
    return -1, disp_title, genres, record


def main():
    parser = argparse.ArgumentParser(description="Predict a rating for one movie.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--title", help="Movie title (substring match, case-insensitive)")
    group.add_argument("--movie-id", type=int, help="Exact MovieLens movieId")
    parser.add_argument(
        "--year",
        type=int,
        help="Release year -- REQUIRED with --title (disambiguates MovieLens substring "
        "matches and any TMDB search fallback); unused with --movie-id",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="Short user name (e.g. 'adeeb') or exact data/ folder name -- whose "
        "ratings to fold in (see engine/paths.resolve_user_dir)",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Use hybrid_mf_artifacts_<tag>.npz (e.g. from a --tag sweep run) instead of the default model",
    )
    args = parser.parse_args()
    if args.title is not None and args.year is None:
        parser.error("--year is required when using --title")
    user_dir = resolve_user_dir(args.user)

    artifacts_name = f"hybrid_mf_artifacts_{args.model_tag}.npz" if args.model_tag else "hybrid_mf_artifacts.npz"
    artifacts = load_hybrid_artifacts(DATA_DIR / artifacts_name)
    meta = load_movie_metadata(DATA_DIR / "ml-32m" / "movies.csv")
    # movieId -> tmdbId, for fetching a cold-but-in-MovieLens movie's content
    # live without a fuzzy title search (we already know exactly which film
    # it is). Drops the ~124 links.csv rows with no tmdbId, same as its
    # tmdbId -> movieId direction does.
    movieid_to_tmdbid = {v: k for k, v in load_tmdb_to_ml(DATA_DIR / "ml-32m" / "links.csv").items()}

    # --- Resolve title/id to a single movieId ---
    fresh_record = None  # set only when a live TMDB record was already fetched during resolution
    if args.movie_id is not None:
        movie_id = args.movie_id
        if movie_id not in meta:
            print(f"movieId {movie_id} not found in ml-32m/movies.csv.")
            sys.exit(1)
        title, genres = meta[movie_id]
    else:
        candidates = find_movie(args.title, args.year, meta)
        if not candidates:
            # No MovieLens entry for this title+year -- try a live TMDB
            # search instead of giving up (e.g. a movie released after this
            # ml-32m snapshot, or one of the minority of titles with no
            # parsed year at all).
            movie_id, title, genres, fresh_record = resolve_via_live_tmdb(args.title, args.year)
        elif len(candidates) > 1:
            print(f"{len(candidates)} titles match {args.title!r} ({args.year}) -- re-run with --movie-id:")
            for mid, t, g in candidates[:20]:
                print(f"  {mid:8d}  {t}  [{g}]")
            if len(candidates) > 20:
                print(f"  ... and {len(candidates) - 20} more")
            sys.exit(1)
        else:
            movie_id, title, genres = candidates[0]

    # --- Fold in the user's ratings ---
    # Iterate over Series columns directly (not itertuples()) -- yields Any,
    # so int()/str() resolve cleanly without Scalar-type complaints.
    matched_df = pd.read_csv(user_dir / "movielens_matched.csv")
    already_rated = {
        int(mid): float(rating)
        for mid, rating in zip(matched_df["movieId"], matched_df["rating"])
    }
    matched = [{"movieId": mid, "rating": rating} for mid, rating in already_rated.items()]
    # Uses fold_in_user's own default alpha (100.0, as of 2026-09-09) --
    # backed by a real leave-one-out CV sweep, not a one-off case; see
    # fold_in_user's docstring (engine/recommend.py) for the full rationale.
    user_vector, user_bias = fold_in_user(matched, artifacts)

    # --- Score: warm (trained item vector) vs. cold (live fetch either way) ---
    idx = artifacts["movieid_to_idx"].get(movie_id)
    if idx is not None:
        item_vector = artifacts["item_embeddings"][idx]
        item_bias = float(artifacts["item_biases"][idx])
        status_label = "warm (trained item vector)"
    else:
        # Cold, either way: no MovieLens id at all (fresh_record already
        # fetched during resolution above), or a MovieLens id that's just
        # under MIN_RATINGS (fetch it live now via its authoritative
        # links.csv tmdbId -- no fuzzy search needed, we already know
        # exactly which film it is). Both score identically from here.
        if fresh_record is None:
            tmdb_id = movieid_to_tmdbid.get(movie_id)
            if tmdb_id is None:
                print(
                    f"{title!r} (movieId {movie_id}) has no trained embedding and no "
                    "tmdbId in ml-32m/links.csv either -- can't fetch content to score."
                )
                sys.exit(1)
            api_key = get_tmdb_api_key()
            fresh_record = asyncio.run(fetch_movie_record(tmdb_id, api_key))
            if fresh_record is None:
                print(f"{title!r} (movieId {movie_id}, tmdb_id {tmdb_id}) -- live TMDB fetch failed.")
                sys.exit(1)
            status_label = "cold (MovieLens entry, <20 ratings -- live TMDB fetch)"
        else:
            status_label = "cold (no MovieLens entry -- live TMDB fetch)"

        content_store = load_content_store(
            DATA_DIR / "tmdb_content_features.npz",
            DATA_DIR / "tmdb_content_features_sparse.npz",
            DATA_DIR / "tmdb_content_tfidf.pkl",
        )
        feat_row = get_content_row(-1, content_store, fresh_record)
        item_vector, item_bias, _ = get_item_representation(-1, artifacts, feat_row)

    raw_pred = predict_rating(user_vector, user_bias, item_vector, item_bias, artifacts["global_mean"])
    clamped = max(0.5, min(5.0, raw_pred))
    print_prediction(title, genres, movie_id, status_label, raw_pred, clamped, already_rated)


if __name__ == "__main__":
    main()
