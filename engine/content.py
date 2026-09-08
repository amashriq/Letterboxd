# Content-side utilities for the hybrid MF recommender.
#
# Loads the trained hybrid MF artifacts (item embeddings + content-feature
# embeddings from scripts/train_hybrid_mf.py) and the TMDB content vocabulary
# (from scripts/fetch_tmdb_content.py), and produces a content-feature row
# for ANY movie -- one already in the training corpus (cheap lookup) or a
# brand-new one with no trained item embedding at all (live TMDB fetch +
# encode). That second case is what lets a movie absent from MovieLens still
# get a real score: the same "identity embedding + content term" formula
# HybridMF uses for every warm item, just with the identity term omitted
# since there is no row for it to look up.
#
# Public API:
#   artifacts     = load_hybrid_artifacts(Path("data/lightfm_artifacts.npz"))
#   content_store = load_content_store(
#       Path("data/tmdb_content_features.npz"),
#       Path("data/tmdb_content_features_sparse.npz"),
#       Path("data/tmdb_content_tfidf.pkl"),
#   )
#   record   = await fetch_movie_record(tmdb_id, api_key)          # only for movies not already in content_store
#   feat_row = get_content_row(movie_id, content_store, record)    # None if genuinely unencodable
#   item_vector, item_bias, is_cold = get_item_representation(movie_id, artifacts, feat_row)

import asyncio
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse

# Allow importing sibling scripts/ modules without a package install step.
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.match import RateLimiter
from scripts.fetch_tmdb_content import GENRE_ID_TO_IDX, _fetch_one


# ---------------------------------------------------------------------------
# Loaders (call once at startup)
# ---------------------------------------------------------------------------

def load_hybrid_artifacts(npz_path: "Path | str") -> dict:
    """
    Load the trained hybrid MF artifacts produced by
    ``scripts/train_hybrid_mf.py``.

    Returns
    -------
    dict with keys:
        - ``item_embeddings``            -- (n_items, k)
        - ``item_biases``                -- (n_items,)
        - ``content_feature_embeddings`` -- (k, n_features)
        - ``content_feature_biases``     -- (n_features,)
        - ``movie_ids``                  -- (n_items,), dtype int
        - ``feature_names``              -- (n_features,)
        - ``global_mean``                -- Python float, mean rating over the
          training population (see engine.recommend.fold_in_user)
        - ``movieid_to_idx``             -- dict[int, int], O(1) movieId → row
    """
    data = np.load(npz_path, allow_pickle=False)
    movie_ids = data["movie_ids"].astype(int)
    return {
        "item_embeddings":            data["item_embeddings"],
        "item_biases":                data["item_biases"],
        "content_feature_embeddings": data["content_feature_embeddings"],
        "content_feature_biases":     data["content_feature_biases"],
        "movie_ids":                  movie_ids,
        "feature_names":              data["feature_names"],
        "global_mean":                float(data["global_mean"]),
        "movieid_to_idx":             {int(mid): i for i, mid in enumerate(movie_ids)},
    }


def load_content_store(
    features_npz_path: "Path | str",
    features_sparse_npz_path: "Path | str",
    tfidf_pkl_path: "Path | str",
) -> dict:
    """
    Load everything needed to get a content-feature row for ANY movie.

    Most movies (anything already in ``tmdb_content_features.npz``) are a
    cheap sparse-matrix row lookup by movieId -- no network call. A movie
    that isn't there yet (released after this corpus was built, or never
    matched to MovieLens) falls through to :func:`encode_movie_content` on a
    freshly-fetched TMDB record.
    """
    meta   = np.load(features_npz_path, allow_pickle=False)
    matrix = sparse.load_npz(str(features_sparse_npz_path))
    with open(tfidf_pkl_path, "rb") as f:
        tfidf = pickle.load(f)

    movie_ids = meta["movie_ids"].astype(int)
    numeric_stats = json.loads(str(meta["numeric_stats_json"][0]))

    return {
        "movieid_to_row": {int(mid): i for i, mid in enumerate(movie_ids)},
        "matrix":         matrix,
        "feature_names":  meta["feature_names"],
        "dir_to_idx":     dict(zip(meta["dir_ids"].tolist(),  meta["dir_idxs"].tolist())),
        "cast_to_idx":    dict(zip(meta["cast_ids"].tolist(), meta["cast_idxs"].tolist())),
        "numeric_stats":  numeric_stats,
        "tfidf":          tfidf,
        "n_kw":           len(tfidf.get_feature_names_out()),
    }


# ---------------------------------------------------------------------------
# Fetching + encoding a movie NOT already in the content store
# ---------------------------------------------------------------------------

async def fetch_movie_record(tmdb_id: int, api_key: str) -> Optional[dict]:
    """
    Fetch one film's /movie, /keywords, /credits from TMDB -- same three
    calls and same record shape as ``scripts/fetch_tmdb_content.py``'s batch
    fetch, just for a single ad-hoc id instead of 87k of them.
    """
    import httpx

    async with httpx.AsyncClient(timeout=20.0) as client:
        return await _fetch_one(
            client,
            asyncio.Semaphore(1),
            RateLimiter(rate=40, period=10.0),
            tmdb_id,
            api_key,
        )


def encode_movie_content(record: dict, vocab: dict) -> np.ndarray:
    """
    Encode one TMDB record (the dict shape ``fetch_movie_record``/
    ``scripts.fetch_tmdb_content._fetch_one`` returns) into a dense feature
    row matching the training corpus's column layout EXACTLY: genre one-hot
    (19) + numeric (7) + keyword TF-IDF + director binary + cast
    billing-weighted -- see ``scripts/fetch_tmdb_content.py:build_features``.

    Uses the vocabulary/normalisation stats persisted at training time
    (``vocab["numeric_stats"]`` etc.) rather than recomputing them -- a
    single new film has no corpus of its own to derive a mean/std from.
    """
    stats = vocab["numeric_stats"]
    n_kw, n_dir, n_cast = vocab["n_kw"], len(vocab["dir_to_idx"]), len(vocab["cast_to_idx"])
    n_features = 19 + 7 + n_kw + n_dir + n_cast
    row = np.zeros(n_features, dtype=np.float32)

    # --- 1. Genre one-hot ---
    for gid in record.get("genre_ids", []):
        col = GENRE_ID_TO_IDX.get(gid)
        if col is not None:
            row[col] = 1.0

    # --- 2. Numeric (same formulas as build_features, saved stats instead of a fit) ---
    vote_avg = float(record.get("vote_average") or 0.0)
    vote_cnt = float(record.get("vote_count") or 0.0)
    runtime  = float(record.get("runtime") or 0.0)
    budget   = float(record.get("budget") or 0.0)
    revenue  = float(record.get("revenue") or 0.0)
    year_str = (record.get("release_date") or "")[:4]

    bayesian_vote = (
        (vote_cnt * vote_avg + stats["bayesian_C"] * stats["bayesian_m"])
        / (vote_cnt + stats["bayesian_C"])
    )
    vote_count_log = np.log1p(vote_cnt)
    runtime_imputed = runtime if runtime > 0 else stats["runtime_impute_value"]
    budget_log  = np.log1p(budget  if budget  > 0 else stats["budget_impute_value"])
    revenue_log = np.log1p(revenue if revenue > 0 else stats["revenue_impute_value"])
    year = int(year_str) if len(year_str) == 4 else 0

    offset = 19
    row[offset + 0] = (bayesian_vote   - stats["bayesian_vote_mean"])  / stats["bayesian_vote_std"]
    row[offset + 1] = (vote_count_log  - stats["vote_count_log_mean"]) / stats["vote_count_log_std"]
    row[offset + 2] = (runtime_imputed - stats["runtime_mean"])        / stats["runtime_std"]
    row[offset + 3] = (budget_log      - stats["budget_log_mean"])     / stats["budget_log_std"]
    row[offset + 4] = (revenue_log     - stats["revenue_log_mean"])    / stats["revenue_log_std"]
    row[offset + 5] = (
        (year - stats["year_mean"]) / (stats["year_std"] + 1e-9) if year > 0 else 0.0
    )
    row[offset + 6] = 1.0 if record.get("original_language") == "en" else 0.0

    # --- 3. Keywords TF-IDF ---
    offset = 19 + 7
    kw_doc = " ".join(record.get("keyword_names", []))
    kw_row = vocab["tfidf"].transform([kw_doc]).toarray().ravel()  # .transform(), NEVER .fit_transform() here
    row[offset : offset + n_kw] = kw_row

    # --- 4. Director binary ---
    offset += n_kw
    for did in record.get("director_ids", []):
        col = vocab["dir_to_idx"].get(did)
        if col is not None:
            row[offset + col] = 1.0

    # --- 5. Cast billing-weighted ---
    offset += n_dir
    for c in record.get("cast", []):
        col = vocab["cast_to_idx"].get(c["id"])
        if col is not None:
            row[offset + col] = 1.0 / (c["order"] + 1)  # lead = 1.0, 2nd = 0.5, …

    return row


def get_content_row(
    movie_id: int,
    content_store: dict,
    fresh_record: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Return a dense content-feature row for ``movie_id``.

    Cheap path: already in the training corpus -- a sparse-matrix row lookup.
    Cold path: pass a TMDB record fetched via ``fetch_movie_record`` for
    ``fresh_record`` and it's encoded on the spot. Returns ``None`` if
    neither is available (movie can't be identified/encoded at all).
    """
    row_idx = content_store["movieid_to_row"].get(int(movie_id))
    if row_idx is not None:
        return content_store["matrix"][row_idx].toarray().ravel().astype(np.float32)
    if fresh_record is not None:
        return encode_movie_content(fresh_record, content_store)
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_item_representation(
    movie_id: int,
    artifacts: dict,
    feat_row: Optional[np.ndarray],
) -> tuple[np.ndarray, float, bool]:
    """
    Return ``(item_vector, item_bias, is_cold)`` for one movie.

    Warm (``movie_id`` has a trained row in ``artifacts``): identity
    embedding, plus the content term on top if ``feat_row`` is supplied --
    exactly ``HybridMF.score``'s ``item_emb_total`` from training.

    Cold (``movie_id`` absent -- never in MovieLens, or filtered out by
    MIN_RATINGS at training time): content term ONLY. There is no identity
    row to fall back to, so ``feat_row`` is required in this case.
    """
    idx = artifacts["movieid_to_idx"].get(int(movie_id))
    feature_embeddings = artifacts["content_feature_embeddings"]  # (k, n_features)
    feature_biases     = artifacts["content_feature_biases"]      # (n_features,)

    if idx is not None:
        item_vector = artifacts["item_embeddings"][idx].copy()
        item_bias   = float(artifacts["item_biases"][idx])
        if feat_row is not None:
            item_vector = item_vector + feat_row @ feature_embeddings.T
            item_bias   = item_bias + float(feat_row @ feature_biases)
        return item_vector, item_bias, False

    if feat_row is None:
        raise ValueError(
            f"movieId {movie_id} has no trained embedding and no content "
            "features were supplied -- cannot score it. Fetch a TMDB record "
            "and pass its encoded row as feat_row."
        )
    item_vector = feat_row @ feature_embeddings.T
    item_bias   = float(feat_row @ feature_biases)
    return item_vector, item_bias, True
