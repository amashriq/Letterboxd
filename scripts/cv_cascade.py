# Cascade sweep: for each candidate latent-factor count k (each its own
# fully-retrained hybrid_mf_artifacts_k<K>.npz, produced separately by
# `train_hybrid_mf.py --components K --tag kK`), cross fold_in_user's Ridge
# alpha against several profile sizes n, to see which (k, alpha) minimizes
# leave-one-out RMSE at each n -- the deliverable is a per-n lookup table
# ("cascade"), not a single global winner.
#
# n is varied by repeatedly subsampling --source-user's warm matched ratings
# (Caroline, n up to ~286) rather than mixing users across n: the same
# person's ratings at different n keeps RMSE comparable across the n axis --
# mixing in a different person per bucket would confound "harder to predict
# person" with "smaller n" (Caroline's own no-personalization RMSE floor is
# ~0.14 higher than Adeeb's -- see the alpha-tuning memory). A single random
# subsample at small n is noisy, so each (k, n) cell is averaged over
# --draws independent random subsamples of that size (n at the full pool
# size only has one possible draws -- no repetition needed there). The SAME
# subsample index sets are reused across every k, generated once up front,
# so the k comparison at a given (n, draw) is apples-to-apples.
#
# --sanity-user (Adeeb, full profile, no subsampling) is scored separately
# as a different-person cross-check on whichever (k, alpha) wins per n --
# not one of the three cascade buckets itself.
#
# Produces:
#   data/cascade-<date>/cascade_results.csv  -- one row per (k, n, alpha): mean/std RMSE over draws
#   data/cascade-<date>/cascade_summary.csv  -- one row per n: the (k, alpha) with lowest mean RMSE
#   data/cascade-<date>/sanity_check.csv     -- one row per (k, alpha) scored on the sanity-user's full profile
#
# Usage:
#   python scripts/cv_cascade.py
#   python scripts/cv_cascade.py --k-values 16,32,64,128,256 --draws 10

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.content import load_hybrid_artifacts
from engine.paths import DATA_DIR, resolve_user_dir
from scripts.cv_alpha import leave_one_out

DEFAULT_K_VALUES = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256]
DEFAULT_ALPHAS = [0.3, 1, 3, 10, 30, 100, 300, 1000]
DEFAULT_N_VALUES = [20, 35, 50, 75, 100, 125, 150, 200, 286]


def load_warm_ratings(user: str, artifacts: dict) -> list[dict]:
    """Load `user`'s matched ratings, filtered to movies `artifacts` has a trained item vector for."""
    user_dir = resolve_user_dir(user)
    matched_df = pd.read_csv(user_dir / "movielens_matched.csv")
    all_ratings = [
        {"movieId": int(mid), "rating": float(rating)}
        for mid, rating in zip(matched_df["movieId"], matched_df["rating"])
    ]
    return [r for r in all_ratings if r["movieId"] in artifacts["movieid_to_idx"]]


def generate_draws(n_values: list[int], pool_size: int, n_draws: int, seed: int) -> dict[int, list[np.ndarray]]:
    """
    Pre-generate, once, the row-index subsamples for every (n, draw) cell --
    reused identically across every k so the k comparison at fixed (n, draw)
    is apples-to-apples. n == pool_size gets exactly one draw (the whole
    pool, in order) since there's only one possible "sample" at full size.
    """
    rng = np.random.default_rng(seed)
    draws: dict[int, list[np.ndarray]] = {}
    for n in n_values:
        if n > pool_size:
            print(f"  [skip] n={n} exceeds pool size {pool_size}")
            continue
        if n == pool_size:
            draws[n] = [np.arange(pool_size)]
        else:
            draws[n] = [rng.choice(pool_size, size=n, replace=False) for _ in range(n_draws)]
    return draws


def main():
    parser = argparse.ArgumentParser(
        description="Sweep latent-factor count k x fold-in alpha across several profile sizes n."
    )
    parser.add_argument("--k-values", default=",".join(map(str, DEFAULT_K_VALUES)))
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--n-values", default=",".join(map(str, DEFAULT_N_VALUES)))
    parser.add_argument("--draws", type=int, default=8, help="Repeated random subsamples per (k, n) cell")
    parser.add_argument("--source-user", default="caroline", help="Pool to subsample n from")
    parser.add_argument("--sanity-user", default="adeeb", help="Full-profile, different-person cross-check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="Output folder (default: data/cascade-<today>/)")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    n_values = [int(n) for n in args.n_values.split(",")]

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR / f"cascade-{pd.Timestamp.today():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load every k's artifacts up front and confirm the warm item set (which
    # movieIds have a trained embedding) is identical across all of them --
    # it's determined by MIN_RATINGS on the shared ml-32m ratings.csv, before
    # any k-specific training, so this should always hold; verify rather than
    # assume, since a mismatch would silently break the shared-draws reuse.
    artifacts_by_k: dict[int, dict] = {}
    for k in k_values:
        path = DATA_DIR / f"hybrid_mf_artifacts_k{k}.npz"
        if not path.exists():
            print(f"  [skip] {path.name} not found -- has that k finished training?")
            continue
        artifacts_by_k[k] = load_hybrid_artifacts(path)

    if not artifacts_by_k:
        sys.exit("No trained k artifacts found -- nothing to evaluate.")

    warm_id_sets = {k: frozenset(a["movieid_to_idx"]) for k, a in artifacts_by_k.items()}
    reference_k = next(iter(warm_id_sets))
    for k, ids in warm_id_sets.items():
        if ids != warm_id_sets[reference_k]:
            print(f"  [warn] k={k}'s warm item set differs from k={reference_k}'s -- draws are no longer "
                  f"strictly apples-to-apples across k for movieIds outside the intersection.")

    source_warm = load_warm_ratings(args.source_user, artifacts_by_k[reference_k])
    sanity_warm = load_warm_ratings(args.sanity_user, artifacts_by_k[reference_k])
    print(f"{args.source_user}: {len(source_warm)} warm ratings (subsampling pool)")
    print(f"{args.sanity_user}: {len(sanity_warm)} warm ratings (sanity check, full profile)")

    draws = generate_draws(n_values, len(source_warm), args.draws, args.seed)

    # --- Main cascade sweep: k x n x alpha, averaged over draws ---
    cascade_rows = []
    for k, artifacts in artifacts_by_k.items():
        for n, idx_lists in draws.items():
            for alpha in alphas:
                rmses = []
                for idx in idx_lists:
                    subset = [source_warm[i] for i in idx]
                    errors = np.array(leave_one_out(subset, artifacts, alpha))
                    rmses.append(float(np.sqrt(np.mean(errors**2))))
                cascade_rows.append({
                    "k": k, "n": n, "alpha": alpha,
                    "mean_rmse": float(np.mean(rmses)), "std_rmse": float(np.std(rmses)),
                    "n_draws": len(rmses),
                })
        print(f"k={k} done")

    cascade_df = pd.DataFrame(cascade_rows)
    cascade_df.to_csv(out_dir / "cascade_results.csv", index=False)
    print(f"\nSaved {len(cascade_df)} rows to {out_dir / 'cascade_results.csv'}")

    # --- Summary: best (k, alpha) per n ---
    summary_rows = []
    for n, group in cascade_df.groupby("n"):
        best = group.loc[group["mean_rmse"].idxmin()]
        summary_rows.append({
            "n": n, "best_k": int(best["k"]), "best_alpha": best["alpha"],
            "best_mean_rmse": best["mean_rmse"], "best_std_rmse": best["std_rmse"],
        })
        print(f"n={n:<4d} best: k={int(best['k']):<4d} alpha={best['alpha']:<7g} "
              f"RMSE={best['mean_rmse']:.4f} +/- {best['std_rmse']:.4f}")
    summary_df = pd.DataFrame(summary_rows).sort_values("n")
    summary_df.to_csv(out_dir / "cascade_summary.csv", index=False)
    print(f"Saved cascade summary to {out_dir / 'cascade_summary.csv'}")

    # --- Sanity check: same (k, alpha) grid scored on a different person's full profile ---
    sanity_rows = []
    for k, artifacts in artifacts_by_k.items():
        for alpha in alphas:
            errors = np.array(leave_one_out(sanity_warm, artifacts, alpha))
            sanity_rows.append({
                "k": k, "alpha": alpha, "rmse": float(np.sqrt(np.mean(errors**2))), "n": len(sanity_warm),
            })
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(out_dir / "sanity_check.csv", index=False)
    print(f"Saved {args.sanity_user} sanity check to {out_dir / 'sanity_check.csv'}")


if __name__ == "__main__":
    main()
