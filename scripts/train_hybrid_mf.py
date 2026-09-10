# Train a hybrid matrix-factorisation model on MovieLens 32M using PyTorch.
# Item embeddings are augmented with a content branch (genres, keywords, directors, cast)
# so cold films can be scored from content features alone at serve time.
#
# Trained via EXPLICIT-rating regression (MSE against real 0.5-5 star ratings),
# not implicit-feedback ranking (BPR/WARP, what LightFM itself offers). BPR only
# optimizes relative order -- "the positive item should outscore a random
# negative" -- it has no notion of a correct absolute value, so its raw score
# is not on the star-rating scale at all. Since the actual product here is a
# predicted rating (a ranked list is a byproduct, not the goal), the model
# needs to be trained on the scale it's meant to predict. This also means
# every example is a single labelled (user, item, rating) triple -- no
# negative sampling needed, unlike a BPR/WARP setup.
#
# Produces:
#   data/hybrid_mf_artifacts.npz  -- item/feature embeddings + movie_ids + feature_names
#   data/hybrid_mf_model.pth      -- full model state dict for fine-tuning or inspection
#   (pass --tag NAME to namespace these + the checkpoint/progress file as
#   hybrid_mf_*_NAME.{npz,pth,json} instead -- e.g. for a components/hparam sweep)
#
# Usage:
#   python scripts/train_hybrid_mf.py                # full run (30 epochs)
#   python scripts/train_hybrid_mf.py --dry-run      # 5 epochs, 1k users (~1 min)
#   python scripts/train_hybrid_mf.py --epochs 50 --components 128
#   python scripts/train_hybrid_mf.py --components 16 --tag k16 --fresh

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing engine from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.paths import DATA_DIR

# Force line-buffered stdout. Python fully buffers stdout by default when it
# isn't attached to a terminal (e.g. redirected to a log file, or run as a
# background job) -- without this, NOTHING prints until the process exits,
# making an 11-hour run impossible to monitor while it's running.
sys.stdout.reconfigure(line_buffering=True)

MIN_RATINGS = 20


def output_paths(tag: "str | None") -> dict[str, Path]:
    """
    Namespace every output file by --tag so parallel/sequential runs at
    different --components (e.g. a latent-factor sweep) don't clobber each
    other's checkpoint/artifacts -- and don't clobber the default (untagged)
    model everything else in this repo reads by default. `tag=None` (or
    unpassed) reproduces the original, untagged filenames exactly.
    """
    suffix = f"_{tag}" if tag else ""
    return {
        "checkpoint": DATA_DIR / f"hybrid_mf_checkpoint{suffix}.pth",
        "progress":   DATA_DIR / f"hybrid_mf_training_progress{suffix}.json",
        "artifacts":  DATA_DIR / f"hybrid_mf_artifacts{suffix}.npz",
        "model":      DATA_DIR / f"hybrid_mf_model{suffix}.pth",
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HybridMF(nn.Module):
    """
    Hybrid matrix factorisation with a content feature branch.

    score(u, i) = dot(user_emb[u], item_emb[i] + feat_row @ feature_embs.T)
                + user_bias[u] + item_bias[i] + feat_row @ feature_biases

    Trained to predict (rating - global_mean) -- see train() -- so add
    global_mean back at serve time (engine.recommend.predict_rating already
    does this).

    Cold items (no interaction history) use only the content terms; warm items
    add the learned item delta on top.

    Architecture based on the LightFM model:
    https://arxiv.org/abs/1507.08439
    """

    def __init__(self, n_users: int, n_items: int, n_features: int, k: int = 64):
        super().__init__()
        # sparse=True: gradients for these tables come back as sparse tensors
        # touching only the rows used in a batch, instead of a dense gradient
        # over the WHOLE table. Paired with SparseAdam in train() below --
        # https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html
        # Without this, backward()/optimizer.step() scale with n_users/n_items
        # (200k+ rows) instead of the batch size (~2k rows) on every step.
        self.user_embeddings    = nn.Embedding(n_users, k, sparse=True)
        self.user_biases        = nn.Embedding(n_users, 1, sparse=True)
        self.item_embeddings    = nn.Embedding(n_items, k, sparse=True)
        self.item_biases        = nn.Embedding(n_items, 1, sparse=True)
        # (k × n_features) — right-multiply by a (batch × n_features) row tensor
        self.feature_embeddings = nn.Parameter(torch.empty(k, n_features))
        self.feature_biases     = nn.Parameter(torch.zeros(n_features))

        # Xavier uniform for all weight tensors
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.feature_embeddings)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def score(
        self,
        user_idxs: torch.Tensor,
        item_idxs: torch.Tensor,
        feat_rows: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        user_idxs : (B,) long
        item_idxs : (B,) long
        feat_rows : (B, n_features) float — dense content row for each item

        Returns
        -------
        (B,) float predicted (rating - global_mean)
        """
        # Content contribution to item embedding and bias
        content_emb  = feat_rows @ self.feature_embeddings.T     # (B, k)
        content_bias = feat_rows @ self.feature_biases            # (B,)

        item_emb_total  = self.item_embeddings(item_idxs) + content_emb          # (B, k)
        item_bias_total = self.item_biases(item_idxs).squeeze(-1) + content_bias # (B,)

        dot = (self.user_embeddings(user_idxs) * item_emb_total).sum(-1)         # (B,)
        return dot + self.user_biases(user_idxs).squeeze(-1) + item_bias_total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dry_run: bool) -> tuple[
    np.ndarray,                     # user_idxs  (n_interactions,) int32
    np.ndarray,                     # item_idxs  (n_interactions,) int32
    np.ndarray,                     # ratings    (n_interactions,) float32
    list[int],                      # sorted_movie_ids
    dict[int, int],                 # movie_id_to_idx
    int,                            # n_users
    float,                          # global_mean rating
]:
    print("Loading ratings...")
    ratings = pd.read_csv(
        DATA_DIR / "ml-32m" / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
    )

    counts      = ratings["movieId"].value_counts()
    valid_movies = counts[counts >= MIN_RATINGS].index
    ratings     = ratings[ratings["movieId"].isin(valid_movies)]
    print(f"  Movies >= {MIN_RATINGS} ratings: {len(valid_movies):,}")
    print(f"  Ratings after filter:           {len(ratings):,}")

    # Global mean rating -- the model is trained to predict the DEVIATION from
    # this, not the raw rating (see train()); engine.recommend.predict_rating
    # adds it back at serve time. Computed on the general population, not any
    # one user, so it's stable across whoever gets folded in later.
    global_mean = float(ratings["rating"].mean())
    print(f"  Global mean rating: {global_mean:.3f}")

    # NOTE: personal ratings are intentionally NOT injected as a trained user
    # here. Every user -- you included -- gets a vector via fold-in
    # (engine.recommend.fold_in_user, Ridge regression against the frozen
    # item vectors trained below) instead of being baked into this training
    # run. That's what makes fold-in actually useful: a new/updated
    # ratings.csv gets scored in milliseconds, not via an 11-hour retrain.

    if dry_run:
        # Keep only first 1,000 unique users to stay fast
        sample_users = ratings["userId"].unique()[:1000].tolist()
        ratings = ratings[ratings["userId"].isin(sample_users)]
        print(f"  --dry-run: trimmed to {ratings['userId'].nunique():,} users")

    sorted_movie_ids: list[int] = sorted(ratings["movieId"].unique().tolist())
    sorted_user_ids: list[int]  = sorted(ratings["userId"].unique().tolist())
    movie_id_to_idx = {mid: i for i, mid in enumerate(sorted_movie_ids)}
    user_id_to_idx  = {uid: i for i, uid in enumerate(sorted_user_ids)}
    n_users = len(sorted_user_ids)

    # Vectorised id->idx mapping + NumPy storage, NOT a Python list of tuples.
    # A list of 31.7M (int, int, float) tuples costs ~150-200 bytes per tuple
    # once Python's per-object overhead and uncached ints are counted --
    # ~5-6GB for the full run, invisible in a 1,000-user dry run (~150K
    # tuples, ~25MB) but enough to OOM-kill the real thing. int32/float32
    # NumPy arrays cost 4 bytes/element with no per-element overhead: the
    # same 31.7M rows drop to ~380MB total.
    user_idxs = ratings["userId"].map(user_id_to_idx).to_numpy(dtype=np.int32)
    item_idxs = ratings["movieId"].map(movie_id_to_idx).to_numpy(dtype=np.int32)
    rating_vals = ratings["rating"].to_numpy(dtype=np.float32)

    print(f"  Users: {n_users:,}  Items: {len(sorted_movie_ids):,}  "
          f"Interactions: {len(user_idxs):,}")
    return user_idxs, item_idxs, rating_vals, sorted_movie_ids, movie_id_to_idx, n_users, global_mean


# ---------------------------------------------------------------------------
# Content matrix alignment
# ---------------------------------------------------------------------------

def build_item_content(
    sorted_movie_ids: list[int],
    movie_id_to_idx: dict[int, int],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Return a (n_items × n_features) sparse matrix aligned to sorted_movie_ids.
    Items with no TMDB coverage get an all-zero row.
    """
    print("Aligning content feature matrix...")
    content_meta   = np.load(DATA_DIR / "tmdb_content_features.npz", allow_pickle=False)
    content_sparse: sparse.csr_matrix = sparse.load_npz(
        str(DATA_DIR / "tmdb_content_features_sparse.npz")
    )
    content_movie_ids = content_meta["movie_ids"].astype(int)
    feature_names     = content_meta["feature_names"]
    n_features        = content_sparse.shape[1]
    n_items           = len(sorted_movie_ids)

    content_row_by_movieid = {int(mid): i for i, mid in enumerate(content_movie_ids)}

    # Stack rows in item-index order; missing rows → zero vectors
    row_blocks = []
    n_covered  = 0
    for movie_id in sorted_movie_ids:
        src = content_row_by_movieid.get(int(movie_id))
        if src is not None:
            row_blocks.append(content_sparse[src])
            n_covered += 1
        else:
            row_blocks.append(sparse.csr_matrix((1, n_features), dtype=np.float32))

    item_content: sparse.csr_matrix = sparse.vstack(row_blocks, format="csr")
    print(f"  Content coverage: {n_covered}/{n_items} items "
          f"({n_covered / n_items * 100:.1f}%)")
    return item_content, feature_names


# ---------------------------------------------------------------------------
# RMSE evaluation
# ---------------------------------------------------------------------------

def compute_rmse(
    model: HybridMF,
    user_idxs: np.ndarray,
    item_idxs: np.ndarray,
    ratings: np.ndarray,
    item_content_dense: torch.Tensor,
    global_mean: float,
    n_sample: int = 200_000,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Sample interactions and compute RMSE between predicted and actual rating,
    on the real 0.5-5 scale (adds global_mean back). Not a held-out metric --
    like the training set itself, this is a training-fit progress signal, not
    a generalisation estimate; a proper train/test split is a separate,
    later evaluation step.
    """
    model.eval()
    rng = np.random.default_rng(42)
    sample = rng.choice(len(user_idxs), size=min(n_sample, len(user_idxs)), replace=False)

    u_t   = torch.from_numpy(user_idxs[sample]).long().to(device)
    i_t   = torch.from_numpy(item_idxs[sample]).long().to(device)
    y_t   = torch.from_numpy(ratings[sample]).float().to(device)
    feats = item_content_dense[i_t]

    with torch.no_grad():
        pred = model.score(u_t, i_t, feats) + global_mean

    rmse = torch.sqrt(F.mse_loss(pred, y_t)).item()
    model.train()
    return rmse


# ---------------------------------------------------------------------------
# Checkpointing -- resumable training state, separate from the final
# hybrid_mf_artifacts.npz/hybrid_mf_model.pth serving format. A long unattended
# run has real ways to die mid-training (forced OS update/restart, dead
# battery during sleep, a closed terminal) with nothing recoverable unless
# progress is saved incrementally, not just once at the very end.
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    epoch: int,
    model: HybridMF,
    sparse_optimizer: torch.optim.Optimizer,
    dense_optimizer: torch.optim.Optimizer,
    components: int,
    dry_run: bool,
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict":            model.state_dict(),
        "sparse_optimizer_state_dict": sparse_optimizer.state_dict(),
        "dense_optimizer_state_dict":  dense_optimizer.state_dict(),
        "components": components,
        "dry_run":    dry_run,
    }, path)


def save_artifacts(
    out_npz: Path,
    out_pth: Path,
    model: HybridMF,
    sorted_movie_ids: list[int],
    feature_names: np.ndarray,
    global_mean: float,
) -> None:
    """Write the servable artifacts -- refreshed after EVERY epoch (see
    train()) so there's always an up-to-date, usable model on disk, not just
    a resumable checkpoint, even if the run dies mid-training."""
    np.savez(
        out_npz,
        item_embeddings=model.item_embeddings.weight.detach().cpu().numpy(),
        item_biases=model.item_biases.weight.detach().cpu().numpy().squeeze(-1),
        content_feature_embeddings=model.feature_embeddings.detach().cpu().numpy(),
        content_feature_biases=model.feature_biases.detach().cpu().numpy(),
        movie_ids=np.array(sorted_movie_ids, dtype=np.int64),
        feature_names=feature_names,
        global_mean=np.float64(global_mean),
    )
    torch.save(model.state_dict(), out_pth)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = output_paths(args.tag)
    checkpoint_path, progress_path = paths["checkpoint"], paths["progress"]
    out_npz, out_pth = paths["artifacts"], paths["model"]

    user_idxs, item_idxs, rating_vals, sorted_movie_ids, movie_id_to_idx, n_users, global_mean = load_data(
        dry_run=args.dry_run
    )
    n_interactions = len(user_idxs)
    item_content, feature_names = build_item_content(sorted_movie_ids, movie_id_to_idx)

    n_items    = len(sorted_movie_ids)
    n_features = item_content.shape[1]
    n_epochs   = args.epochs if not args.dry_run else 5

    print(f"\nModel: {n_users} users | {n_items} items | {n_features} features | k={args.components}")

    # Precompute the dense content matrix ONCE and keep it resident on `device`.
    # The per-batch alternative -- item_content[idxs].toarray() -- re-slices the
    # sparse matrix and allocates+fills a fresh dense array from scratch on
    # every single batch; on a CUDA device it would also mean a fresh
    # host->device copy every batch. At full scale (23,350 items x 6,248
    # features) the dense matrix is only ~584MB, so building it once and
    # indexing into it with plain tensor gathers is far cheaper than repeating
    # the sparse->dense conversion ~15k times/epoch.
    print("Caching dense content matrix on device...")
    item_content_dense = torch.tensor(
        item_content.toarray(), dtype=torch.float32, device=device
    )
    print(f"  {item_content_dense.shape[0]:,} items x {item_content_dense.shape[1]:,} features "
          f"(~{item_content_dense.numel() * 4 / 1e6:.0f} MB)")

    model = HybridMF(n_users, n_items, n_features, k=args.components).to(device)

    # Two optimisers: the (sparse=True) embedding tables need SparseAdam,
    # which only touches rows with nonzero gradients; the content-branch
    # matrices are small (k x n_features) and dense, so regular Adam is fine.
    # weight_decay (L2) on the dense optimiser only -- SparseAdam doesn't
    # support it. This matters more here than it would for BPR: the content
    # branch sums one embedding per active feature (the LightFM "feature-sum"
    # mechanism), so a richly-tagged film sums more terms than a sparsely
    # tagged one; without a regularising pull back toward zero, nothing stops
    # that sum from growing with tag count rather than staying on the rating
    # scale.
    # https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    sparse_params = (
        list(model.user_embeddings.parameters())
        + list(model.user_biases.parameters())
        + list(model.item_embeddings.parameters())
        + list(model.item_biases.parameters())
    )
    dense_params = [model.feature_embeddings, model.feature_biases]
    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=args.lr)
    dense_optimizer  = torch.optim.Adam(dense_params, lr=args.lr, weight_decay=args.weight_decay)

    # Resume from a checkpoint unless --fresh was passed or none exists.
    # A mismatched --components/--dry-run against the checkpoint means the
    # saved tensors are the wrong shape to load into this run's model --
    # fail loudly rather than silently training the wrong thing.
    start_epoch = 1
    if checkpoint_path.exists() and not args.fresh:
        print(f"Found checkpoint at {checkpoint_path.name}, resuming...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if ckpt["components"] != args.components or ckpt["dry_run"] != args.dry_run:
            raise ValueError(
                f"Checkpoint was --components={ckpt['components']} dry_run={ckpt['dry_run']}, "
                f"this run is --components={args.components} dry_run={args.dry_run}. "
                "Pass --fresh to discard the checkpoint and start over, or match its settings."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        sparse_optimizer.load_state_dict(ckpt["sparse_optimizer_state_dict"])
        dense_optimizer.load_state_dict(ckpt["dense_optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}/{n_epochs}")

    rng = np.random.default_rng(0)

    # Shuffle index buffer, allocated ONCE outside the epoch loop and
    # reshuffled in place every epoch (rng.shuffle mutates, it doesn't
    # allocate). The first version of this loop instead did
    # `user_idxs[perm]` -- a FRESH full-size (127MB x 3 + 254MB perm) copy
    # every single epoch. Freed memory should get reused by the allocator,
    # but Windows' default allocator fragments under exactly this pattern
    # (many large alloc/free cycles), so the process's footprint crept up
    # epoch over epoch instead of staying flat -- confirmed by an actual OOM
    # kill at epoch 5. Slicing `perm[start:end]` below is a cheap VIEW (plain
    # slicing, not fancy indexing); only the resulting small
    # (batch_size,) gathers from user_idxs/item_idxs/rating_vals allocate
    # anything, and those are freed every batch, not accumulated.
    perm = np.arange(n_interactions, dtype=np.int64)

    # RMSE is the second-most expensive step -- no need to pay for it every epoch.
    eval_every = 1 if args.dry_run else args.eval_every

    for epoch in range(start_epoch, n_epochs + 1):
        rng.shuffle(perm)  # in place -- no new allocation

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_interactions, args.batch_size):
            end = start + args.batch_size
            batch_idx = perm[start:end]  # view, not a copy

            u_t = torch.from_numpy(user_idxs[batch_idx]).long().to(device)
            i_t = torch.from_numpy(item_idxs[batch_idx]).long().to(device)
            y_t = torch.from_numpy(rating_vals[batch_idx] - global_mean).float().to(device)

            # Plain tensor gather out of the cached dense matrix -- see note above.
            feats = item_content_dense[i_t]

            pred = model.score(u_t, i_t, feats)
            loss = F.mse_loss(pred, y_t)

            sparse_optimizer.zero_grad()
            dense_optimizer.zero_grad()
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        do_eval  = (epoch % eval_every == 0) or (epoch == n_epochs)
        rmse = None
        rmse_str = ""
        if do_eval:
            rmse = compute_rmse(
                model, user_idxs, item_idxs, rating_vals, item_content_dense,
                global_mean, device=device,
            )
            rmse_str = f"  RMSE: {rmse:.4f}"
        print(f"Epoch {epoch:3d}/{n_epochs}  MSE loss: {avg_loss:.4f}{rmse_str}")

        # Checkpoint + refresh the servable artifacts EVERY epoch, not just at
        # the end -- see save_checkpoint()'s docstring for why. Cheap: a
        # ~15-20 min epoch can easily afford a few seconds of disk I/O.
        save_checkpoint(
            checkpoint_path, epoch, model, sparse_optimizer, dense_optimizer,
            args.components, args.dry_run,
        )
        save_artifacts(out_npz, out_pth, model, sorted_movie_ids, feature_names, global_mean)
        progress_path.write_text(json.dumps({
            "epoch": epoch, "of": n_epochs,
            "mse_loss": avg_loss, "rmse": rmse,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "done": epoch == n_epochs,
        }, indent=2))

    print(f"\nTraining complete. Saved:")
    print(f"  {out_npz.name}  -- {n_items} items × {args.components} dims + content weights")
    print(f"  {out_pth.name}  -- model state dict")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hybrid MF model on MovieLens 32M with PyTorch, "
                     "via explicit-rating regression (MSE)."
    )
    parser.add_argument("--dry-run",     action="store_true",
                        help="1,000 users, 5 epochs — quick smoke test (~1 min)")
    parser.add_argument("--fresh",       action="store_true",
                        help="Ignore any existing checkpoint and start from epoch 1.")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--components",  type=int,   default=64,  dest="components")
    parser.add_argument("--batch-size",  type=int,   default=2048)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="L2 regularisation on the content-branch weights "
                             "(feature_embeddings/feature_biases). Default: 1e-5.")
    parser.add_argument("--eval-every",  type=int,   default=5,
                        help="Compute RMSE every N epochs (always at the last). Default: 5.")
    parser.add_argument("--tag",         type=str,   default=None,
                        help="Namespace all output files as hybrid_mf_*_<tag>.{npz,pth,json} "
                             "instead of the default untagged names -- use for a latent-factor/"
                             "hyperparameter sweep so runs don't clobber each other or the "
                             "default model. Also namespaces checkpoint resume.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
