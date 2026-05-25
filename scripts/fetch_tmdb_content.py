# Batch-fetches TMDB metadata for every MovieLens film and builds a content
# feature matrix for use in LightFM hybrid model training.
#
# Calls 3 TMDB endpoints per film:
#   GET /movie/{id}          -- genres, runtime, year, language, ratings, budget, revenue
#   GET /movie/{id}/keywords -- keyword tags (e.g. "heist", "based on novel")
#   GET /movie/{id}/credits  -- top-5 cast (billing-weighted) + director
#
# Checkpoints raw JSON to data/tmdb_content_raw.jsonl so the run can be
# interrupted and resumed without losing progress.
#
# Produces:
#   data/tmdb_content_features.npz   -- scipy sparse feature matrix + vocabulary
#   data/tmdb_content_tfidf.pkl      -- fitted TF-IDF transformer (for encoding
#                                      new films at serve time)
#
# Usage:
#   python scripts/fetch_tmdb_content.py            # full run (~5 hrs)
#   python scripts/fetch_tmdb_content.py --limit 50 # dry run first 50 films

import argparse
import asyncio
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Allow importing engine from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.match import RateLimiter

load_dotenv()

TMDB_BASE = "https://api.themoviedb.org/3"
DATA_DIR   = Path(__file__).parent.parent / "data"
CHECKPOINT = DATA_DIR / "tmdb_content_raw.jsonl"
OUT_NPZ    = DATA_DIR / "tmdb_content_features.npz"
OUT_TFIDF  = DATA_DIR / "tmdb_content_tfidf.pkl"

# Stable TMDB movie genre IDs → column index in the one-hot block.
# https://developer.themoviedb.org/reference/genre-movie-list
GENRE_ID_TO_IDX: dict[int, int] = {
    28: 0,     # Action
    12: 1,     # Adventure
    16: 2,     # Animation
    35: 3,     # Comedy
    80: 4,     # Crime
    99: 5,     # Documentary
    18: 6,     # Drama
    10751: 7,  # Family
    14: 8,     # Fantasy
    36: 9,     # History
    27: 10,    # Horror
    10402: 11, # Music
    9648: 12,  # Mystery
    10749: 13, # Romance
    878: 14,   # Science Fiction
    10770: 15, # TV Movie
    53: 16,    # Thriller
    10752: 17, # War
    37: 18,    # Western
}
N_GENRES = 19
GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Sci-Fi", "TV Movie", "Thriller", "War", "Western",
]


# ---------------------------------------------------------------------------
# Async fetch layer
# ---------------------------------------------------------------------------

async def _get(
    client: httpx.AsyncClient,
    concurrency: asyncio.Semaphore,
    rate: RateLimiter,
    url: str,
    params: dict,
) -> dict | None:
    """Single rate-limited GET, returns parsed JSON or None on error."""
    async with concurrency:
        async with rate:
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                print(f"  [warn] {url}: {exc}")
                return None


async def _fetch_one(
    client: httpx.AsyncClient,
    concurrency: asyncio.Semaphore,
    rate: RateLimiter,
    tmdb_id: int,
    api_key: str,
) -> dict | None:
    """
    Fetch /movie/{id}, /movie/{id}/keywords, /movie/{id}/credits in parallel.
    Returns a combined record dict, or None if the main endpoint fails.
    """
    p = {"api_key": api_key}
    base = f"{TMDB_BASE}/movie/{tmdb_id}"

    detail, kw_resp, cr_resp = await asyncio.gather(
        _get(client, concurrency, rate, base,             p),
        _get(client, concurrency, rate, f"{base}/keywords", p),
        _get(client, concurrency, rate, f"{base}/credits",  p),
    )

    if detail is None:
        return None

    return {
        "tmdb_id":           tmdb_id,
        "genre_ids":         [g["id"]   for g in detail.get("genres", [])],
        "runtime":           detail.get("runtime") or 0,
        "release_date":      detail.get("release_date", ""),
        "original_language": detail.get("original_language", ""),
        "vote_average":      detail.get("vote_average", 0.0),
        "vote_count":        detail.get("vote_count",   0),
        "budget":            detail.get("budget",  0),
        "revenue":           detail.get("revenue", 0),
        "keyword_ids":   [k["id"]   for k in (kw_resp or {}).get("keywords", [])],
        "keyword_names": [k["name"] for k in (kw_resp or {}).get("keywords", [])],
        "director_ids":  [
            c["id"] for c in (cr_resp or {}).get("crew", [])
            if c.get("job") == "Director"
        ],
        "cast": [
            {"id": c["id"], "order": c["order"]}
            for c in (cr_resp or {}).get("cast", [])[:5]  # top-5 billed
        ],
    }


async def fetch_all(
    tmdb_ids: list[int],
    api_key: str,
    limit: int | None = None,
    rate_limit: int = 40,
    rate_period: float = 10.0,
    max_concurrent: int = 10,
    timeout: float = 20.0,
) -> dict[int, dict]:
    """
    Async batch fetch for all TMDB IDs.

    Resumes from ``data/tmdb_content_raw.jsonl`` if it exists -- already-fetched
    IDs are skipped. New results are appended to the same file as they arrive.
    """
    # --- Load checkpoint ---
    fetched: dict[int, dict] = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    fetched[rec["tmdb_id"]] = rec
        print(f"  Resuming -- {len(fetched)} records already in checkpoint.")

    remaining = [tid for tid in tmdb_ids if tid not in fetched]
    if limit is not None:
        remaining = remaining[:limit]

    if not remaining:
        print("  Nothing to fetch -- all IDs already checkpointed.")
        return fetched

    print(f"  Fetching {len(remaining)} films "
          f"({len(fetched)} already in checkpoint, "
          f"{len(tmdb_ids) - len(fetched) - len(remaining)} no TMDB ID)...")

    limiter     = RateLimiter(rate_limit, rate_period)
    concurrency = asyncio.Semaphore(max_concurrent)
    done        = 0

    async def fetch_and_write(tid: int, f_out) -> dict | None:
        nonlocal done
        result = await _fetch_one(client, concurrency, limiter, tid, api_key)
        done += 1
        if result is not None:
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
        if done % 500 == 0 or done == len(remaining):
            pct = done / len(remaining) * 100
            print(f"  Progress: {done}/{len(remaining)} ({pct:.1f}%)")
        return result

    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(CHECKPOINT, "a", encoding="utf-8") as f_out:
            results = await asyncio.gather(
                *[fetch_and_write(tid, f_out) for tid in remaining]
            )

    for tid, result in zip(remaining, results):
        if result is not None:
            fetched[tid] = result

    print(f"  Fetch complete. {len(fetched)} total records.")
    return fetched


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _znorm(x: np.ndarray) -> np.ndarray:
    """Z-score normalise, safe against zero std."""
    std = x.std()
    return (x - x.mean()) / (std if std > 1e-9 else 1.0)


def build_features(
    records: dict[int, dict],
    movieid_by_tmdbid: dict[int, int],
) -> dict:
    """
    Convert raw per-film dicts into a combined sparse feature matrix.

    Feature blocks (in column order):
      1. Genre one-hot        (19 dims)
      2. Numeric              ( 7 dims): vote_average (Bayesian), vote_count (log),
                                          runtime, budget (log), revenue (log),
                                          release_year, is_english
      3. Keywords TF-IDF      (~500–2000 dims, min_df=5)
      4. Director binary      (~300–800 dims, min film count=3)
      5. Cast billing-weighted (~800–2000 dims, min film count=5)

    Returns a dict with the matrix, vocabulary, and stats needed for serving.
    """
    # Only keep records that have a MovieLens ID
    tmdb_ids = [tid for tid in records if tid in movieid_by_tmdbid]
    movie_ids = [movieid_by_tmdbid[tid] for tid in tmdb_ids]
    recs = [records[tid] for tid in tmdb_ids]
    n = len(recs)
    print(f"  Building features for {n} films...")

    # ── 1. Genre one-hot ──────────────────────────────────────────────────
    genre_rows, genre_cols, genre_data = [], [], []
    for i, rec in enumerate(recs):
        for gid in rec["genre_ids"]:
            col = GENRE_ID_TO_IDX.get(gid)
            if col is not None:
                genre_rows.append(i); genre_cols.append(col); genre_data.append(1.0)
    genre_sp = sparse.csr_matrix(
        (genre_data, (genre_rows, genre_cols)), shape=(n, N_GENRES), dtype=np.float32
    )

    # ── 2. Numeric features ──────────────────────────────────────────────
    vote_avg   = np.array([r["vote_average"] for r in recs], dtype=np.float64)
    vote_cnt   = np.array([r["vote_count"]   for r in recs], dtype=np.float64)
    runtimes   = np.array([r["runtime"]      for r in recs], dtype=np.float64)
    budgets    = np.array([r["budget"]       for r in recs], dtype=np.float64)
    revenues   = np.array([r["revenue"]      for r in recs], dtype=np.float64)
    years      = np.array([
        int(r["release_date"][:4]) if len(r["release_date"]) >= 4 else 0
        for r in recs
    ], dtype=np.float64)
    is_english = np.array([1.0 if r["original_language"] == "en" else 0.0 for r in recs])

    # Bayesian vote average: pull toward global mean when vote_count is low
    # https://www.fxsolver.com/browse/formulas/Bayesian+average
    C = 100.0
    rated_mask = vote_cnt > 0
    m = vote_avg[rated_mask].mean() if rated_mask.any() else 6.0
    bayesian_vote = (vote_cnt * vote_avg + C * m) / (vote_cnt + C)

    # Impute zeros with median before log-scaling
    def log_impute(arr: np.ndarray) -> np.ndarray:
        pos = arr[arr > 0]
        med = float(np.median(pos)) if len(pos) > 0 else 1.0
        return np.log1p(np.where(arr > 0, arr, med))

    def year_norm(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
        valid = arr[arr > 0]
        mu  = float(valid.mean())  if len(valid) > 0 else 1990.0
        std = float(valid.std())   if len(valid) > 0 else 20.0
        return np.where(arr > 0, (arr - mu) / (std + 1e-9), 0.0), mu, std

    runtime_imputed = np.where(runtimes > 0, runtimes, runtimes[runtimes > 0].mean() if (runtimes > 0).any() else 90.0)
    years_norm, year_mean, year_std = year_norm(years)

    numeric_dense = np.column_stack([
        _znorm(bayesian_vote),
        _znorm(np.log1p(vote_cnt)),
        _znorm(runtime_imputed),
        _znorm(log_impute(budgets)),
        _znorm(log_impute(revenues)),
        years_norm,
        is_english,
    ]).astype(np.float32)
    numeric_sp = sparse.csr_matrix(numeric_dense)

    numeric_stats = {
        "bayesian_C": C,
        "bayesian_m": float(m),
        "year_mean":  float(year_mean),
        "year_std":   float(year_std),
    }

    # ── 3. Keywords TF-IDF ───────────────────────────────────────────────
    # Each film's keywords treated as a "document" of space-separated tokens.
    # min_df=5 for large corpora; falls back to min_df=1 for small dry runs.
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    kw_docs  = [" ".join(r["keyword_names"]) for r in recs]
    min_df   = min(5, max(1, n // 200))   # 1 for n<200, scales up to 5 at n≥1000
    tfidf    = TfidfVectorizer(min_df=min_df, max_features=2000, sublinear_tf=True)
    try:
        kw_sp = sparse.csr_matrix(tfidf.fit_transform(kw_docs), dtype=np.float32)
    except ValueError:
        # All keywords pruned (very small corpus) -- fall back to min_df=1
        tfidf = TfidfVectorizer(min_df=1, max_features=2000, sublinear_tf=True)
        kw_sp = sparse.csr_matrix(tfidf.fit_transform(kw_docs), dtype=np.float32)
    kw_feature_names = [f"kw:{w}" for w in tfidf.get_feature_names_out()]

    # ── 4. Director binary ───────────────────────────────────────────────
    dir_counter: Counter = Counter()
    for rec in recs:
        for did in rec["director_ids"]:
            dir_counter[did] += 1
    freq_directors = [did for did, cnt in dir_counter.items() if cnt >= 3]
    dir_to_idx     = {did: i for i, did in enumerate(freq_directors)}
    n_dirs         = len(dir_to_idx)

    dr_rows, dr_cols, dr_data = [], [], []
    for i, rec in enumerate(recs):
        for did in rec["director_ids"]:
            if did in dir_to_idx:
                dr_rows.append(i); dr_cols.append(dir_to_idx[did]); dr_data.append(1.0)
    dir_sp = sparse.csr_matrix(
        (dr_data, (dr_rows, dr_cols)), shape=(n, n_dirs), dtype=np.float32
    )
    dir_feature_names = [f"director:{did}" for did in freq_directors]

    # ── 5. Cast billing-weighted ─────────────────────────────────────────
    cast_counter: Counter = Counter()
    for rec in recs:
        for c in rec["cast"]:
            cast_counter[c["id"]] += 1
    freq_cast  = [cid for cid, cnt in cast_counter.items() if cnt >= 5]
    cast_to_idx = {cid: i for i, cid in enumerate(freq_cast)}
    n_cast      = len(cast_to_idx)

    ca_rows, ca_cols, ca_data = [], [], []
    for i, rec in enumerate(recs):
        for c in rec["cast"]:
            if c["id"] in cast_to_idx:
                weight = 1.0 / (c["order"] + 1)  # lead = 1.0, 2nd = 0.5, …
                ca_rows.append(i); ca_cols.append(cast_to_idx[c["id"]]); ca_data.append(weight)
    cast_sp = sparse.csr_matrix(
        (ca_data, (ca_rows, ca_cols)), shape=(n, n_cast), dtype=np.float32
    )
    cast_feature_names = [f"cast:{cid}" for cid in freq_cast]

    # ── Combine ───────────────────────────────────────────────────────────
    # cast() tells Pylance the concrete type; sparse.hstack return type is a
    # broad union that Pylance can't narrow despite format="csr".
    feature_matrix = cast(sparse.csr_matrix, sparse.hstack(
        [genre_sp, numeric_sp, kw_sp, dir_sp, cast_sp], format="csr"
    ))

    genre_feature_names = [f"genre:{name}" for name in GENRE_NAMES]
    numeric_feature_names = [
        "vote_average_bayesian", "vote_count_log", "runtime",
        "budget_log", "revenue_log", "release_year", "is_english",
    ]
    all_feature_names = (
        genre_feature_names + numeric_feature_names
        + kw_feature_names + dir_feature_names + cast_feature_names
    )

    n_films, n_feats = feature_matrix.shape  # type: ignore[misc]
    _, n_kw = kw_sp.shape                    # type: ignore[misc]
    print(f"  Feature matrix: {n_films} films x {n_feats} features "
          f"({feature_matrix.nnz / (n_films * n_feats) * 100:.2f}% dense)")
    print(f"    Genres: {N_GENRES}  Numerics: 7  Keywords: {n_kw}  "
          f"Directors: {n_dirs}  Cast: {n_cast}")

    return {
        "movie_ids":          np.array(movie_ids, dtype=np.int64),
        "tmdb_ids":           np.array(tmdb_ids,  dtype=np.int64),
        "feature_matrix":     feature_matrix,
        "feature_names":      np.array(all_feature_names),
        "tfidf":              tfidf,          # save separately as pickle
        "dir_to_idx":         dir_to_idx,
        "cast_to_idx":        cast_to_idx,
        "numeric_stats":      numeric_stats,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_features(result: dict) -> None:
    """Save sparse matrix (scipy .npz) and TF-IDF transformer (pickle)."""
    fm = result["feature_matrix"]

    # scipy sparse → separate arrays in a numpy .npz
    # (np.savez doesn't support sparse directly)
    sparse.save_npz(str(OUT_NPZ).replace(".npz", "_sparse.npz"), fm)

    # Metadata alongside: movieids, tmdbids, feature_names, vocab info
    np.savez(
        OUT_NPZ,
        movie_ids=result["movie_ids"],
        tmdb_ids=result["tmdb_ids"],
        feature_names=result["feature_names"],
        dir_ids=np.array(list(result["dir_to_idx"].keys()),  dtype=np.int64),
        dir_idxs=np.array(list(result["dir_to_idx"].values()), dtype=np.int32),
        cast_ids=np.array(list(result["cast_to_idx"].keys()),  dtype=np.int64),
        cast_idxs=np.array(list(result["cast_to_idx"].values()), dtype=np.int32),
        numeric_stats_json=np.array([json.dumps(result["numeric_stats"])]),
    )

    with open(OUT_TFIDF, "wb") as f:
        pickle.dump(result["tfidf"], f)

    print(f"\nSaved:")
    print(f"  {OUT_NPZ.name}           -- metadata + vocabulary")
    print(f"  {OUT_NPZ.name.replace('.npz', '_sparse.npz')}    -- {fm.shape[0]}×{fm.shape[1]} feature matrix")
    print(f"  {OUT_TFIDF.name}          -- TF-IDF transformer")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-fetch TMDB content features.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N films (for dry runs).")
    parser.add_argument("--skip-build", action="store_true",
                        help="Fetch only; skip feature matrix build (resume later).")
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise ValueError("Set TMDB_API_KEY in .env or environment.")

    # Load movieId ↔ tmdbId mapping from links.csv
    links_path = DATA_DIR / "ml-32m" / "links.csv"
    links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"]).dropna(subset=["tmdbId"])
    links["tmdbId"]  = links["tmdbId"].astype(int)
    links["movieId"] = links["movieId"].astype(int)
    movieid_by_tmdbid = dict(zip(links["tmdbId"], links["movieId"]))
    tmdb_ids = links["tmdbId"].tolist()
    print(f"Links loaded: {len(tmdb_ids)} films with TMDB IDs.")

    # Fetch
    records = asyncio.run(fetch_all(tmdb_ids, api_key, limit=args.limit))

    if args.skip_build:
        print("--skip-build: skipping feature matrix. Re-run without flag to build.")
        return

    # Build and save
    result = build_features(records, movieid_by_tmdbid)
    save_features(result)


if __name__ == "__main__":
    main()
