# One-off evaluation script: holds out a random slice of MovieLens
# interactions, trains a HybridMF model (reusing the exact model class and
# content-alignment logic from scripts/train_hybrid_mf.py, unmodified) on
# the rest, and reports held-out RMSE/MAE three ways on the same test set:
#
#   hybrid         -- identity + content, exactly as trained (the real model)
#   identity_only  -- content term zeroed out (plain biased MF, no TMDB signal)
#   content_only   -- identity term zeroed out -- this is EXACTLY the formula
#                      engine/content.py:get_item_representation uses to score
#                      a cold item, so it measures cold-item accuracy using
#                      held-out ratings on items the model has ground truth
#                      for, rather than being a hypothetical.
#
# This is a post-hoc ablation of ONE jointly-trained model (not three
# separately-trained ones) -- deliberately, since content_only reproduces
# the exact weights production cold-item scoring actually uses, and this
# avoids confounding the comparison with separate-run training variance.
#
# Does not touch scripts/train_hybrid_mf.py or the production
# hybrid_mf_artifacts.npz. The production model was trained on 100% of the
# data with no held-out split (see its own module docstring), so there's no
# way to retroactively get a fair generalization number from it -- this
# script trains a SEPARATE, smaller-scale model (see --n-users/--epochs)
# purely to get one. Reduced scale is a feasibility tradeoff (the full
# pipeline's ~200K users would take as long as the original training run),
# not a claim that this matches the production model's exact numbers.
#
# Usage:
#   python scripts/eval_holdout.py --n-users 20000 --epochs 12

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.paths import DATA_DIR
from scripts.train_hybrid_mf import MIN_RATINGS, HybridMF, build_item_content

# Same fix as train_hybrid_mf.py's, for the same reason: stdout is fully
# buffered (not line-buffered) when it isn't attached to a terminal, so a
# backgrounded run's progress is invisible in its log file until the
# process exits without this.
sys.stdout.reconfigure(line_buffering=True)


def load_split(n_users: "int | None", test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random row-level train/test split of MovieLens ratings.csv, filtered to
    the same MIN_RATINGS>=20 items the production pipeline trains on.
    Test rows whose user or movie doesn't survive into the train split
    (no trained embedding to evaluate against) are dropped -- rare at this
    filter threshold, but not impossible for a user with very few ratings.
    """
    ratings = pd.read_csv(DATA_DIR / "ml-32m" / "ratings.csv", usecols=["userId", "movieId", "rating"])
    counts = ratings["movieId"].value_counts()
    valid_movies = counts[counts >= MIN_RATINGS].index
    ratings = ratings[ratings["movieId"].isin(valid_movies)]

    rng = np.random.default_rng(seed)
    if n_users is not None:
        all_users = ratings["userId"].unique()
        sample_users = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)
        ratings = ratings[ratings["userId"].isin(sample_users)]

    is_test = rng.random(len(ratings)) < test_frac
    train = ratings[~is_test].copy()
    test = ratings[is_test].copy()

    train_users = set(train["userId"].unique())
    train_movies = set(train["movieId"].unique())
    before = len(test)
    test = test[test["userId"].isin(train_users) & test["movieId"].isin(train_movies)]
    dropped = before - len(test)
    if dropped:
        print(f"  Dropped {dropped} test rows whose user/movie has no training rows left")

    return train, test


def main():
    parser = argparse.ArgumentParser(
        description="Held-out RMSE + identity/content ablation for the hybrid MF model (reduced-scale, standalone run)."
    )
    parser.add_argument("--n-users", type=int, default=20000,
                         help="Subsample this many users for a feasible run (production trains on all ~200K)")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading + splitting ratings...")
    train, test = load_split(args.n_users, args.test_frac, args.seed)
    print(f"  Train interactions: {len(train):,}   Test interactions: {len(test):,}")

    sorted_movie_ids = sorted(train["movieId"].unique().tolist())
    sorted_user_ids = sorted(train["userId"].unique().tolist())
    movie_id_to_idx = {mid: i for i, mid in enumerate(sorted_movie_ids)}
    user_id_to_idx = {uid: i for i, uid in enumerate(sorted_user_ids)}
    n_users_, n_items = len(sorted_user_ids), len(sorted_movie_ids)
    global_mean = float(train["rating"].mean())
    print(f"  Users: {n_users_:,}  Items: {n_items:,}  global_mean={global_mean:.3f}")

    item_content, feature_names = build_item_content(sorted_movie_ids, movie_id_to_idx)
    n_features = item_content.shape[1]
    item_content_dense = torch.tensor(item_content.toarray(), dtype=torch.float32, device=device)

    def to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # .copy(): .map()'s output can hand back a non-writable view, which
        # torch.from_numpy warns about (harmless here -- nothing mutates
        # these arrays in place -- but the copy is cheap and silences it).
        u = df["userId"].map(user_id_to_idx).to_numpy(dtype=np.int64).copy()
        i = df["movieId"].map(movie_id_to_idx).to_numpy(dtype=np.int64).copy()
        r = df["rating"].to_numpy(dtype=np.float32).copy()
        return u, i, r

    train_u, train_i, train_r = to_arrays(train)
    test_u, test_i, test_r = to_arrays(test)

    model = HybridMF(n_users_, n_items, n_features, k=args.components).to(device)
    sparse_params = (
        list(model.user_embeddings.parameters()) + list(model.user_biases.parameters())
        + list(model.item_embeddings.parameters()) + list(model.item_biases.parameters())
    )
    dense_params = [model.feature_embeddings, model.feature_biases]
    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=args.lr)
    dense_optimizer = torch.optim.Adam(dense_params, lr=args.lr, weight_decay=args.weight_decay)

    rng = np.random.default_rng(args.seed)
    perm = np.arange(len(train_u), dtype=np.int64)

    print("Training...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(perm)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            u_t = torch.from_numpy(train_u[batch_idx]).to(device)
            i_t = torch.from_numpy(train_i[batch_idx]).to(device)
            y_t = torch.from_numpy(train_r[batch_idx] - global_mean).float().to(device)
            feats = item_content_dense[i_t]

            pred = model.score(u_t, i_t, feats)
            loss = F.mse_loss(pred, y_t)

            sparse_optimizer.zero_grad()
            dense_optimizer.zero_grad()
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        print(f"  Epoch {epoch:3d}/{args.epochs}  MSE loss: {epoch_loss / max(n_batches, 1):.4f}")
    elapsed = time.time() - t0
    print(f"Training done in {elapsed / 60:.1f} min")

    # --- Held-out evaluation, three ways, all from the SAME trained model ---
    # Batched, same as training -- materializing item_content_dense[i_t] for
    # the WHOLE test set at once (a first version of this did exactly that)
    # allocates an (n_test x n_features) dense tensor -- at full scale,
    # 3.17M x ~6.2K float32 is ~74GB, an instant CUDA OOM. Accumulate sum of
    # squared/absolute error per batch instead; mathematically identical to
    # computing RMSE/MAE over the full tensor at once.
    model.eval()
    eval_batch_size = args.batch_size * 4  # no gradients to track -- can afford a bigger batch than training
    variant_names = ("hybrid", "identity_only", "content_only")
    sq_err_sum = {name: 0.0 for name in variant_names}
    abs_err_sum = {name: 0.0 for name in variant_names}
    n_test = len(test_r)

    with torch.no_grad():
        for start in range(0, n_test, eval_batch_size):
            end = start + eval_batch_size
            u_t = torch.from_numpy(test_u[start:end]).to(device)
            i_t = torch.from_numpy(test_i[start:end]).to(device)
            y_t = torch.from_numpy(test_r[start:end]).float().to(device)
            feats = item_content_dense[i_t]

            content_emb = feats @ model.feature_embeddings.T
            content_bias = feats @ model.feature_biases
            item_emb_id = model.item_embeddings(i_t)
            item_bias_id = model.item_biases(i_t).squeeze(-1)
            user_emb = model.user_embeddings(u_t)
            user_bias = model.user_biases(u_t).squeeze(-1)

            preds = {
                # hybrid: identity + content, exactly as trained
                "hybrid": (user_emb * (item_emb_id + content_emb)).sum(-1) + user_bias + item_bias_id + content_bias,
                # identity_only: content term zeroed -- plain biased MF
                "identity_only": (user_emb * item_emb_id).sum(-1) + user_bias + item_bias_id,
                # content_only: identity term zeroed -- exact cold-item formula
                "content_only": (user_emb * content_emb).sum(-1) + user_bias + content_bias,
            }
            for name, pred in preds.items():
                err = (pred + global_mean) - y_t
                sq_err_sum[name] += float((err ** 2).sum())
                abs_err_sum[name] += float(err.abs().sum())

    results = []
    for name in variant_names:
        rmse = (sq_err_sum[name] / n_test) ** 0.5
        mae = abs_err_sum[name] / n_test
        results.append({"variant": name, "rmse": rmse, "mae": mae, "n_test": n_test})
        print(f"  {name:15s}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    out_path = Path(args.out) if args.out else DATA_DIR / f"holdout_eval_{pd.Timestamp.today():%Y-%m-%d}.csv"
    results_df = pd.DataFrame(results)
    results_df["n_users"] = n_users_
    results_df["n_items"] = n_items
    results_df["epochs"] = args.epochs
    results_df["k"] = args.components
    results_df["train_interactions"] = len(train_u)
    results_df["elapsed_min"] = round(elapsed / 60, 2)
    results_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
