from .content import (
    fetch_movie_record,
    get_content_row,
    get_item_representation,
    load_content_store,
    load_hybrid_artifacts,
    search_tmdb_by_title,
)
from .match import RateLimiter, load_tmdb_to_ml, match_ratings, pick_best_tmdb_match, resolve_tmdb_ids
from .paths import DATA_DIR, resolve_user_dir
from .recommend import fold_in_user, load_movie_metadata, predict_rating

__all__ = [
    # engine.content
    "fetch_movie_record",
    "get_content_row",
    "get_item_representation",
    "load_content_store",
    "load_hybrid_artifacts",
    "search_tmdb_by_title",
    # engine.match
    "RateLimiter",
    "load_tmdb_to_ml",
    "match_ratings",
    "pick_best_tmdb_match",
    "resolve_tmdb_ids",
    # engine.paths
    "DATA_DIR",
    "resolve_user_dir",
    # engine.recommend (current hybrid-MF API only -- see engine/recommend.py's
    # header for the legacy SVD-era functions this deliberately omits)
    "fold_in_user",
    "load_movie_metadata",
    "predict_rating",
]
