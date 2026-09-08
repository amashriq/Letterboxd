# Data Directory

## Source Files (from Letterboxd export)
- **ratings.csv** — Raw Letterboxd export. Personal movie ratings with title, year, and URI.
- **diary.csv** — Raw Letterboxd export. Every logged watch including rewatches and unrated films.

## Generated Files
- **tmdb_metadata.csv** — Personal ratings enriched with TMDB IDs. Produced by `scripts/fetch_tmdb_metadata.py`.
- **movielens_matched.csv** — Subset of tmdb_metadata.csv for movies that exist in the MovieLens dataset, with the MovieLens `movieId` added. Produced by `scripts/match_movielens.py`.
- **movielens_unmatched.csv** — Subset of tmdb_metadata.csv for movies with no match in MovieLens (typically recent releases). Produced by `scripts/match_movielens.py`.

- **svd_item_vectors.npz** — Item embedding matrix from Funk SVD trained on MovieLens 32M. Contains `item_vectors` (n_items × k), `item_biases`, `movieids`, and `global_mean`. Produced by `scripts/archive/train_svd.py` (archived — SVD pipeline superseded by PyTorch).
- **svd_user_vector.npz** — Personal user vector from SVD joint training. Contains `user_vector`, `user_bias`. Produced by `scripts/archive/train_svd.py` (archived).

- **tmdb_content_features.npz** — TMDB metadata arrays for all MovieLens films: movieids, tmdbids, feature names, genre/cast/director lookup tables. Produced by `scripts/fetch_tmdb_content.py`.
- **tmdb_content_features_sparse.npz** — Sparse feature matrix (scipy CSR) for all MovieLens films. Columns: genre one-hot (19) + numerics (7) + keywords TF-IDF + director binary + cast billing-weighted. Produced by `scripts/fetch_tmdb_content.py`.
- **tmdb_content_tfidf.pkl** — Fitted TfidfVectorizer for keyword encoding at serve time. Produced by `scripts/fetch_tmdb_content.py`.
- **tmdb_content_raw.jsonl** — Checkpoint file from TMDB batch fetch (one JSON record per film, used for resumable runs). Produced by `scripts/fetch_tmdb_content.py`.

- **hybrid_mf_artifacts.npz** — Trained hybrid MF embeddings, refreshed after every epoch (not just at the end). Keys: `item_embeddings` (n_items × k), `item_biases` (n_items,), `content_feature_embeddings` (k × n_features), `content_feature_biases` (n_features,), `movie_ids` (n_items,), `feature_names` (n_features,), `global_mean` (scalar — model predicts rating deviation from this; add it back at serve time). Produced by `scripts/train_hybrid_mf.py`.
- **hybrid_mf_model.pth** — Full PyTorch model state dict for fine-tuning or inspection, refreshed every epoch. Produced by `scripts/train_hybrid_mf.py`.
- **hybrid_mf_checkpoint.pth** — Resumable training state (model + both optimizers' state dicts + epoch number), overwritten every epoch so an interrupted run can continue with `python scripts/train_hybrid_mf.py` (same args) instead of restarting; pass `--fresh` to ignore it. Not needed for serving. Produced by `scripts/train_hybrid_mf.py`.
- **hybrid_mf_training_progress.json** — `{epoch, of, mse_loss, rmse, updated_at, done}`, overwritten every epoch — cheap way to check training progress without loading the checkpoint. Produced by `scripts/train_hybrid_mf.py`.

## Directories
- **ml-32m/** — MovieLens 32M dataset. Downloaded from [grouplens.org](https://grouplens.org/datasets/movielens/).
  - **ratings.csv** — Raw user↔movie ratings (`userId`, `movieId`, `rating`, `timestamp`); ~32M rows, 0.5–5.0 star scale. Consumed by `scripts/train_hybrid_mf.py`.
- **extra/** — Additional Letterboxd export files (watchlist, likes, reviews, etc.).
