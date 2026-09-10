# Leave-one-out cross-validation of fold_in_user's Ridge alpha: for a single
# user's matched ratings, hold out each warm rating in turn, refit the user
# vector on the rest, and score how well it predicts the held-out rating --
# across several candidate alpha values -- to pick a value backed by
# held-out error rather than the one-off case in fold_in_user's docstring.
#
# Only ratings on movies with a trained item vector ("warm", i.e. present in
# hybrid_mf_artifacts.npz) are used, both as fold-in inputs and as held-out
# targets -- fold_in_user already silently drops non-warm ratings from its
# training data, and scoring a held-out cold item would mix in the TMDB
# content-embedding/correction machinery, confounding the alpha comparison.
#
# Produces:
#   data/<user>-<date>/alpha_cv_results.csv -- one row per alpha tried
#
# Usage:
#   python scripts/cv_alpha.py --user adeeb
#   python scripts/cv_alpha.py --user adeeb --alphas 1,5,10,15,20,50

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.content import load_hybrid_artifacts
from engine.paths import DATA_DIR, resolve_user_dir
from engine.recommend import fold_in_user, predict_rating


def leave_one_out(warm_ratings: list[dict], artifacts: dict, alpha: float) -> list[float]:
    """Return one (clamped_prediction - actual) error per warm rating, held out in turn."""
    errors = []
    for i in range(len(warm_ratings)):
        train = warm_ratings[:i] + warm_ratings[i + 1:]
        held_out = warm_ratings[i]
        user_vector, user_bias = fold_in_user(train, artifacts, alpha=alpha)

        idx = artifacts["movieid_to_idx"][held_out["movieId"]]
        item_vector = artifacts["item_embeddings"][idx]
        item_bias = float(artifacts["item_biases"][idx])

        raw_pred = predict_rating(user_vector, user_bias, item_vector, item_bias, artifacts["global_mean"])
        clamped = max(0.5, min(5.0, raw_pred))
        errors.append(clamped - held_out["rating"])
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-out CV of fold_in_user's Ridge alpha on one user's matched ratings."
    )
    parser.add_argument("--user", required=True, help="Short user name (e.g. 'adeeb') or exact data/ folder name")
    parser.add_argument(
        "--alphas",
        default="1,5,10,15,20",
        help="Comma-separated alpha values to try (default: 1,5,10,15,20)",
    )
    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    user_dir = resolve_user_dir(args.user)
    artifacts = load_hybrid_artifacts(DATA_DIR / "hybrid_mf_artifacts.npz")

    matched_df = pd.read_csv(user_dir / "movielens_matched.csv")
    all_ratings = [
        {"movieId": int(mid), "rating": float(rating)}
        for mid, rating in zip(matched_df["movieId"], matched_df["rating"])
    ]
    warm_ratings = [r for r in all_ratings if r["movieId"] in artifacts["movieid_to_idx"]]

    n_dropped = len(all_ratings) - len(warm_ratings)
    print(
        f"{len(all_ratings)} matched ratings, {len(warm_ratings)} warm (trained item vector) "
        f"-- {n_dropped} dropped (below MIN_RATINGS at training time)."
    )
    if len(warm_ratings) < 3:
        print("Fewer than 3 warm ratings -- not enough to leave one out and still fit Ridge. Aborting.")
        sys.exit(1)

    results = []
    for alpha in alphas:
        errors = np.array(leave_one_out(warm_ratings, artifacts, alpha))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        mean_error = float(np.mean(errors))  # signed -- systematic over/under-prediction
        results.append({"alpha": alpha, "rmse": rmse, "mae": mae, "mean_error": mean_error, "n": len(errors)})
        print(f"alpha={alpha:<6g} RMSE={rmse:.4f}  MAE={mae:.4f}  mean_error={mean_error:+.4f}  n={len(errors)}")

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["rmse"].idxmin()]
    print(f"\nBest by RMSE: alpha={best['alpha']:g} (RMSE={best['rmse']:.4f})")

    out_path = user_dir / "alpha_cv_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
