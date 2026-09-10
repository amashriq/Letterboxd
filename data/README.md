# Data Directory

## Per-User Data — `data/<user>-<date>/`

Each Letterboxd profile used by the pipeline gets its own dated folder, e.g.
`data/alice-2026-09-08/`, `data/bob-2026-09-09/` (the date is when the
export was pulled in, not necessarily when Letterboxd generated it). Scripts
that operate on personal ratings take a required `--user` flag (short name,
e.g. `alice`, resolving to that user's most recent dated folder; or an exact
folder name to target a specific dated snapshot instead — see
`engine/paths.resolve_user_dir`).

**Source files (raw Letterboxd export), per folder:**
- **ratings.csv** — Personal movie ratings with title, year, and URI.
- **diary.csv** — Every logged watch including rewatches and unrated films.
- **comments.csv**, **profile.csv**, **reviews.csv**, **watched.csv**, **watchlist.csv** — Other raw export files; not currently consumed by any script.
- **likes/**, **deleted/**, **orphaned/** — Present in some exports only (Letterboxd includes these depending on account history/settings); not currently consumed by any script.

**Generated files, per folder:**
- **tmdb_metadata.csv** — That user's ratings enriched with TMDB IDs. Columns include `letterboxd_year` (the original Letterboxd `Year`, kept for auditing) alongside TMDB's own `release_year` — the TMDB match is picked by proximity to `letterboxd_year` (within 1 year, to allow for a festival-vs-wide-release discrepancy on the same film), not just TMDB's search ranking, since blindly trusting the top search result mismatches real title collisions (e.g. Mean Girls 2004 vs. 2024, The Lion King 1994 vs. 2019). A title with no TMDB result within a plausible year is dropped to unmatched rather than guessed. Produced by `scripts/fetch_tmdb_metadata.py --user <name>`.
- **movielens_matched.csv** — Subset of tmdb_metadata.csv for movies that exist in the MovieLens dataset, with the MovieLens `movieId` added. Also drops any `tmdb_id` that maps to more than one `movieId` in `ml-32m/links.csv` (a small number of ambiguous ids in that file) to `movielens_unmatched.csv` rather than picking one arbitrarily. Produced by `scripts/match_movielens.py --user <name>`.
- **movielens_unmatched.csv** — Subset of tmdb_metadata.csv for movies with no match in MovieLens (typically recent releases), plus any rows dropped for an ambiguous `links.csv` mapping (see above). Produced by `scripts/match_movielens.py --user <name>`.
- **alpha_cv_results.csv** — Leave-one-out CV of `fold_in_user`'s Ridge `alpha` against that user's warm matched ratings: one row per alpha tried (`alpha`, `rmse`, `mae`, `mean_error`, `n`). Produced by `scripts/cv_alpha.py --user <name>`.

## Generated Files (shared, at `data/` root)
- **svd_item_vectors.npz** *(not currently present in `data/`)* — Item embedding matrix from Funk SVD trained on MovieLens 32M. Contains `item_vectors` (n_items × k), `item_biases`, `movieids`, and `global_mean`. Produced by `scripts/archive/train_svd.py` (archived — SVD pipeline superseded by PyTorch); re-run that script to regenerate if needed.
- **svd_user_vector.npz** *(not currently present in `data/`)* — Personal user vector from SVD joint training. Contains `user_vector`, `user_bias`. Produced by `scripts/archive/train_svd.py` (archived).

- **tmdb_content_features.npz** — TMDB metadata arrays for all MovieLens films: movieids, tmdbids, feature names, genre/cast/director lookup tables. Produced by `scripts/fetch_tmdb_content.py`.
- **tmdb_content_features_sparse.npz** — Sparse feature matrix (scipy CSR) for all MovieLens films. Columns: genre one-hot (19) + numerics (7) + keywords TF-IDF + director binary + cast billing-weighted. Produced by `scripts/fetch_tmdb_content.py`.
- **tmdb_content_tfidf.pkl** — Fitted TfidfVectorizer for keyword encoding at serve time. Produced by `scripts/fetch_tmdb_content.py`.
- **tmdb_content_raw.jsonl** — Checkpoint file from TMDB batch fetch (one JSON record per film, used for resumable runs). Produced by `scripts/fetch_tmdb_content.py`.

- **hybrid_mf_artifacts.npz** — Trained hybrid MF embeddings, refreshed after every epoch (not just at the end). Keys: `item_embeddings` (n_items × k), `item_biases` (n_items,), `content_feature_embeddings` (k × n_features), `content_feature_biases` (n_features,), `movie_ids` (n_items,), `feature_names` (n_features,), `global_mean` (scalar — model predicts rating deviation from this; add it back at serve time). Produced by `scripts/train_hybrid_mf.py`.
- **hybrid_mf_model.pth** — Full PyTorch model state dict for fine-tuning or inspection, refreshed every epoch. Produced by `scripts/train_hybrid_mf.py`.
- **hybrid_mf_checkpoint.pth** — Resumable training state (model + both optimizers' state dicts + epoch number), overwritten every epoch so an interrupted run can continue with `python scripts/train_hybrid_mf.py` (same args) instead of restarting; pass `--fresh` to ignore it. Not needed for serving. Produced by `scripts/train_hybrid_mf.py`.
- **hybrid_mf_training_progress.json** — `{epoch, of, mse_loss, rmse, updated_at, done}`, overwritten every epoch — cheap way to check training progress without loading the checkpoint. Produced by `scripts/train_hybrid_mf.py`.
- **hybrid_mf_artifacts_k\<K\>.npz / hybrid_mf_model_k\<K\>.pth / hybrid_mf_checkpoint_k\<K\>.pth / hybrid_mf_training_progress_k\<K\>.json** — Same as the four entries above, but for a `--components K --tag kK` run of a latent-factor sweep, namespaced so they don't clobber the default (untagged) model or each other. Produced by `scripts/train_hybrid_mf.py --tag <tag>` (see `output_paths()`).

## `data/cascade-<date>/` — latent-factor x alpha sweep across profile sizes
One dated run of `scripts/cv_cascade.py`, which cross-references several `hybrid_mf_artifacts_k<K>.npz` sweep runs against `fold_in_user`'s Ridge alpha at several profile sizes n (subsampled from one user's warm matched ratings, repeated over multiple random draws per (k, n) cell to average out subsampling noise), plus a different-person full-profile sanity check.
- **cascade_results.csv** — One row per (`k`, `n`, `alpha`): `mean_rmse`/`std_rmse` over the repeated draws, `n_draws`.
- **cascade_summary.csv** — One row per `n`: the (`best_k`, `best_alpha`) minimizing `mean_rmse` at that profile size — the actual cascade lookup table.
- **sanity_check.csv** — One row per (`k`, `alpha`): RMSE on the sanity-user's full profile (no subsampling), to check whether the cascade's conclusions hold for a different person.
- **k24_alpha_extend/**, **k24_alpha_refine/** — Each holds its own `cascade_results.csv`/`cascade_summary.csv`/`sanity_check.csv`, same shape as above but a narrower `--k-values 24 --alphas ...` re-run — matches the "bracketed alpha re-check at k=24" follow-up described in `fold_in_user`'s docstring (`engine/recommend.py`).

## `holdout_eval_<label>.csv` — held-out RMSE + identity/content ablation
One run of `scripts/eval_holdout.py`, which trains a SEPARATE, standalone HybridMF model on a genuine train/test split of `ml-32m/ratings.csv` (the production model in `hybrid_mf_artifacts.npz` was trained on 100% of the data with no split, so there's no way to get a fair generalization number from it directly). One row per variant (`hybrid`, `identity_only`, `content_only` — see the script's header comment for what each ablates): `rmse`, `mae`, `n_test`, plus the run's `n_users`/`n_items`/`epochs`/`k`/`train_interactions`/`elapsed_min` for reproducibility. `holdout_eval_full.csv` is the full-scale run (all ~200K users, 30 epochs, matching production's configuration exactly except for the held-out split); `holdout_eval_full.log` is its training log.

## Directories
- **ml-32m/** — MovieLens 32M dataset. Downloaded from [grouplens.org](https://grouplens.org/datasets/movielens/).
  - **ratings.csv** — Raw user↔movie ratings (`userId`, `movieId`, `rating`, `timestamp`); ~32M rows, 0.5–5.0 star scale. Consumed by `scripts/train_hybrid_mf.py`.
