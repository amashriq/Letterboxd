# CLI to recommend the top-N unrated movies for a user's folded-in profile,
# optionally filtered by genre (e.g. rom-coms = movies tagged with ALL of
# Romance, Comedy in ml-32m/movies.csv). Scores every warm item the same way
# predict.py's warm path does -- pure identity embedding + bias, no content
# term -- via the folded-in user vector, excludes already-rated movies, and
# prints the top N by predicted rating.
#
# Usage:
#   python scripts/recommend.py --user caroline --genre Romance,Comedy --top-n 5
#   python scripts/recommend.py --user adeeb --top-n 10
#   python scripts/recommend.py --user caroline --model-tag k24   # use a tagged sweep model

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.content import load_hybrid_artifacts
from engine.paths import DATA_DIR, resolve_user_dir
from engine.recommend import fold_in_user, load_movie_metadata, predict_rating


def main():
    parser = argparse.ArgumentParser(description="Recommend top-N unrated movies for a folded-in user.")
    parser.add_argument("--user", required=True, help="Short user name (e.g. 'adeeb') or exact data/ folder name")
    parser.add_argument("--genre", default=None,
                        help="Comma-separated genre tokens that must ALL appear in the movie's genre string "
                             "(e.g. 'Romance,Comedy' for rom-coms)")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--model-tag", default=None,
                         help="Use hybrid_mf_artifacts_<tag>.npz (e.g. from a --tag sweep run) instead of the default model")
    parser.add_argument("--min-ratings", type=int, default=0,
                         help="Drop candidates with fewer than this many ml-32m ratings -- a thinly-rated item's "
                              "identity embedding is noisier, so an outsized dot-product score there is as likely "
                              "to be overfit as genuine taste match (see fold_in_user's docstring history). "
                              "Default 0 = no floor beyond the MIN_RATINGS=20 already required to be warm at all.")
    parser.add_argument("--min-support-sim", type=float, default=0.0,
                         help="Require a candidate's item embedding to have at least this cosine similarity to "
                              "SOME movie the user actually rated, before trusting its personalization score. "
                              "A large dot-product boost with no corroborating rated movie nearby in embedding "
                              "space is exactly the failure mode this guards against (see the alpha-tuning memory's "
                              "Fast & Furious/Tokyo Drift case -- 0.914 similar to each other, ~0.04-0.11 similar "
                              "to anything the user actually rated highly). Default 0.0 = no filter.")
    args = parser.parse_args()

    artifacts_name = f"hybrid_mf_artifacts_{args.model_tag}.npz" if args.model_tag else "hybrid_mf_artifacts.npz"
    artifacts = load_hybrid_artifacts(DATA_DIR / artifacts_name)
    meta = load_movie_metadata(DATA_DIR / "ml-32m" / "movies.csv")

    rating_counts = None
    if args.min_ratings > 0:
        rating_counts = pd.read_csv(DATA_DIR / "ml-32m" / "ratings.csv", usecols=["movieId"])["movieId"].value_counts()

    user_dir = resolve_user_dir(args.user)
    matched_df = pd.read_csv(user_dir / "movielens_matched.csv")
    already_rated = {int(mid): float(r) for mid, r in zip(matched_df["movieId"], matched_df["rating"])}
    matched = [{"movieId": mid, "rating": r} for mid, r in already_rated.items()]

    user_vector, user_bias = fold_in_user(matched, artifacts)

    # Precompute the user's rated-item embeddings, L2-normalised, once -- used
    # below as the corroborating-evidence pool for --min-support-sim.
    rated_idx = [artifacts["movieid_to_idx"][mid] for mid in already_rated if mid in artifacts["movieid_to_idx"]]
    rated_matrix = artifacts["item_embeddings"][rated_idx]
    rated_norms = np.linalg.norm(rated_matrix, axis=1, keepdims=True)
    rated_normed = rated_matrix / np.maximum(rated_norms, 1e-9)

    def support_similarity(item_vector: np.ndarray) -> float:
        """Max cosine similarity between `item_vector` and any movie the user actually rated."""
        norm = np.linalg.norm(item_vector)
        if norm < 1e-9:
            return 0.0
        sims = (rated_normed @ item_vector) / norm
        return float(sims.max())

    genre_tokens = [g.strip().lower() for g in args.genre.split(",")] if args.genre else []

    scored = []
    for movie_id, idx in artifacts["movieid_to_idx"].items():
        if movie_id in already_rated:
            continue
        title, genres = meta.get(movie_id, (None, ""))
        if title is None:
            continue
        if genre_tokens and not all(tok in genres.lower() for tok in genre_tokens):
            continue
        if rating_counts is not None and rating_counts.get(movie_id, 0) < args.min_ratings:
            continue
        item_vector = artifacts["item_embeddings"][idx]
        if args.min_support_sim > 0:
            support = support_similarity(item_vector)
            if support < args.min_support_sim:
                continue
        else:
            support = None
        item_bias = float(artifacts["item_biases"][idx])
        raw_pred = predict_rating(user_vector, user_bias, item_vector, item_bias, artifacts["global_mean"])
        clamped = max(0.5, min(5.0, raw_pred))
        scored.append((clamped, raw_pred, movie_id, title, genres, support))

    scored.sort(key=lambda row: row[0], reverse=True)
    top = scored[: args.top_n]

    label = f" [{args.genre}]" if args.genre else ""
    print(f"Top {len(top)} recommendations for {args.user}{label} ({len(scored)} candidates scored):")
    for clamped, raw, movie_id, title, genres, support in top:
        support_str = f"  support_sim={support:.3f}" if support is not None else ""
        print(f"  {clamped:.2f}/5.0  {title}  [{genres}]  (movieId {movie_id}, raw {raw:.2f}){support_str}")


if __name__ == "__main__":
    main()
