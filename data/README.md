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

## Directories
- **ml-32m/** — MovieLens 32M dataset. Downloaded from [grouplens.org](https://grouplens.org/datasets/movielens/).
- **extra/** — Additional Letterboxd export files (watchlist, likes, reviews, etc.).
