# Data Directory

## Source Files (from Letterboxd export)
- **ratings.csv** — Raw Letterboxd export. Personal movie ratings with title, year, and URI.
- **diary.csv** — Raw Letterboxd export. Every logged watch including rewatches and unrated films.

## Generated Files
- **tmdb_metadata.csv** — Personal ratings enriched with TMDB IDs. Produced by `scripts/fetch_tmdb_metadata.py`.
- **movielens_matched.csv** — Subset of tmdb_metadata.csv for movies that exist in the MovieLens dataset, with the MovieLens `movieId` added. Produced by `scripts/match_movielens.py`.
- **movielens_unmatched.csv** — Subset of tmdb_metadata.csv for movies with no match in MovieLens (typically recent releases). Produced by `scripts/match_movielens.py`.

## Directories
- **ml-32m/** — MovieLens 32M dataset. Downloaded from [grouplens.org](https://grouplens.org/datasets/movielens/).
- **extra/** — Additional Letterboxd export files (watchlist, likes, reviews, etc.).
