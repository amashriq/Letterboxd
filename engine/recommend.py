# Recommendation engine for the Letterboxd recommender web app.
#
# Loads pre-trained Funk SVD item vectors and scores candidate films for a
# given user based on their Letterboxd-matched ratings.  Supports two tiers
# in this slice (Slice 1):
#
#   0–9 matched ratings  →  popularity scoring  (global_mean + item_bias)
#   10+ matched ratings  →  item-based CF       (weighted cosine similarity)
#
# Fold-in / blend tiers (50–150, 150+) are implemented in Slice 2.
#
# Public API:
#   arts = load_svd_artifacts(Path("data/svd_item_vectors.npz"))
#   meta = load_movie_metadata(Path("data/ml-32m/movies.csv"))
#   recs = recommend(matched, arts, meta, top_n=25, genre_filter="Drama")

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders (call once at startup)
# ---------------------------------------------------------------------------

def load_svd_artifacts(npz_path: "Path | str") -> dict:
    """
    Load pre-trained Funk SVD item vectors from a ``.npz`` archive.

    Call once at application startup and pass the returned dict to every
    scoring function.  The dict is read-only — scoring functions must not
    mutate it.

    Parameters
    ----------
    npz_path:
        Path to ``svd_item_vectors.npz`` produced by ``scripts/train_svd.py``.
        Expected keys: ``item_vectors`` (N×k), ``item_biases`` (N,),
        ``movieids`` (N,), ``global_mean`` (scalar wrapped in a 0-d array).

    Returns
    -------
    dict with keys:
        - ``item_vectors``  — ``np.ndarray`` shape (N, k), float32/64
        - ``item_biases``   — ``np.ndarray`` shape (N,)
        - ``movieids``      — ``np.ndarray`` shape (N,), dtype int
        - ``global_mean``   — Python float
        - ``movieid_to_idx``— ``dict[int, int]``, O(1) movieId → row index
    """
    data = np.load(npz_path)
    movieids = data["movieids"].astype(int)
    return {
        "item_vectors":   data["item_vectors"],
        "item_biases":    data["item_biases"],
        "movieids":       movieids,
        "global_mean":    float(data["global_mean"]),
        "movieid_to_idx": {int(mid): idx for idx, mid in enumerate(movieids)},
    }


def load_movie_metadata(movies_path: "Path | str") -> dict[int, tuple[str, str]]:
    """
    Load MovieLens ``movies.csv`` and return a ``movieId → (title, genres)`` map.

    Genres are the raw pipe-separated string from
    `MovieLens <https://grouplens.org/datasets/movielens/32m/>`_
    (e.g. ``"Action|Adventure|Sci-Fi"``).

    Parameters
    ----------
    movies_path:
        Path to ``ml-32m/movies.csv``.

    Returns
    -------
    dict[int, tuple[str, str]]
        Maps integer MovieLens movie IDs to ``(title, genres)`` tuples.
    """
    df = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
    return {
        int(row.movieId): (row.title, row.genres)
        for row in df.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# Scoring functions (pure — no side effects)
# ---------------------------------------------------------------------------

def _score_popularity(
    artifacts: dict,
    exclude_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score every item by ``global_mean + item_bias``.

    This is the SVD predicted rating for a perfectly average user, so it
    surfaces intrinsically well-liked films regardless of personal taste.
    Used when fewer than 10 ratings have been matched to MovieLens.

    Parameters
    ----------
    artifacts:
        Dict returned by :func:`load_svd_artifacts`.
    exclude_ids:
        Set of MovieLens IDs to suppress (already-rated films).

    Returns
    -------
    scores : np.ndarray, shape (M,)
    indices : np.ndarray, shape (M,)
        Parallel arrays over the M non-excluded items (unsorted).
    """
    movieids = artifacts["movieids"]
    item_biases = artifacts["item_biases"]
    global_mean = artifacts["global_mean"]

    mask = np.array([int(mid) not in exclude_ids for mid in movieids])
    indices = np.where(mask)[0]
    scores = global_mean + item_biases[indices]
    return scores, indices


def _score_item_cf(
    artifacts: dict,
    matched: list[dict],
    exclude_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Item-based collaborative filtering via weighted cosine similarity.

    Builds a user preference vector as a weighted sum of the rated items'
    L2-normalised SVD vectors, where weights are rating deviations from the
    global mean:

    .. math::

        \\mathbf{u} = \\sum_i (r_i - \\mu) \\cdot \\hat{q}_i

    Then scores every candidate ``j`` by:

    .. math::

        \\text{score}(j) = \\hat{q}_j \\cdot \\mathbf{u}

    which is equivalent to ``Σᵢ (rᵢ − μ) · cosine_sim(qᵢ, qⱼ)`` as
    described in the `item-based CF literature
    <https://dl.acm.org/doi/10.1145/371920.372071>`_.

    Parameters
    ----------
    artifacts:
        Dict returned by :func:`load_svd_artifacts`.
    matched:
        Output of ``engine.match.match_ratings`` — each dict must contain
        ``"movieId"`` (int) and ``"rating"`` (float).
    exclude_ids:
        Set of MovieLens IDs to suppress (already-rated films).

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Score for every item in the artifact matrix (zeros for excluded items).
    indices : np.ndarray, shape (N,)
        Corresponding row indices (0 … N-1).
    """
    item_vectors = artifacts["item_vectors"]
    movieid_to_idx = artifacts["movieid_to_idx"]
    global_mean = artifacts["global_mean"]

    # L2-normalise item vectors (local copy — do not mutate artifacts)
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    normed_Q = item_vectors / np.maximum(norms, 1e-9)  # (N, k)

    # Build weighted user preference vector from rated items
    user_pref = np.zeros(item_vectors.shape[1], dtype=np.float64)
    for m in matched:
        idx = movieid_to_idx.get(int(m["movieId"]))
        if idx is None:
            continue  # rated film not in SVD matrix (shouldn't happen)
        deviation = float(m["rating"]) - global_mean
        user_pref += deviation * normed_Q[idx]

    # Score all candidates
    scores = normed_Q @ user_pref  # (N,)

    # Zero out excluded items so they can't surface in top-N
    for mid in exclude_ids:
        idx = movieid_to_idx.get(mid)
        if idx is not None:
            scores[idx] = -np.inf

    indices = np.arange(len(scores))
    return scores, indices


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

# TODO(slice-2): add _score_foldin(artifacts, matched, exclude_ids) using
# RidgeCV to estimate a user vector and compute μ + b_u + b_i + u·qᵢ.
# Blend tiers: 50–150 → item_cf dominant + foldin secondary;
#              150+   → foldin dominant + item_cf secondary.

def recommend(
    matched: list[dict],
    artifacts: dict,
    movie_meta: dict[int, tuple[str, str]],
    top_n: int = 25,
    genre_filter: Optional[str] = None,
) -> list[dict]:
    """
    Return the top-N recommended films for a user.

    Dispatches to the appropriate scoring tier based on how many Letterboxd
    ratings were successfully matched to MovieLens entries:

    * **0–9 matched** → popularity scoring (``global_mean + item_bias``)
    * **10+ matched** → item-based collaborative filtering

    Parameters
    ----------
    matched:
        List of matched-rating dicts from ``engine.match.match_ratings``.
        Each dict must contain ``"movieId"`` and ``"rating"``.
        Pass an empty list if the user has no matched ratings.
    artifacts:
        Dict returned by :func:`load_svd_artifacts`.
    movie_meta:
        Dict returned by :func:`load_movie_metadata`.
    top_n:
        Number of recommendations to return.
    genre_filter:
        Optional genre substring to filter by (case-insensitive).
        E.g. ``"Drama"`` keeps only films whose pipe-separated genres string
        contains ``"drama"``.  If ``None``, no genre filter is applied.

    Returns
    -------
    list[dict]
        Up to ``top_n`` dicts, sorted descending by ``predicted_score``,
        each with keys: ``movieId``, ``title``, ``genres``, ``predicted_score``.
    """
    exclude_ids: set[int] = {int(m["movieId"]) for m in matched}

    # --- Tier dispatch ---
    if len(matched) < 10:
        scores, indices = _score_popularity(artifacts, exclude_ids)
    else:
        scores, indices = _score_item_cf(artifacts, matched, exclude_ids)

    movieids = artifacts["movieids"]

    # --- Genre filter ---
    if genre_filter is not None:
        token = genre_filter.lower()
        genre_mask = np.array(
            [
                token in (movie_meta.get(int(movieids[i]), ("", "(no genres listed)"))[1]).lower()
                for i in indices
            ]
        )
        scores = scores[genre_mask]
        indices = indices[genre_mask]

    # --- Top-N extraction via argpartition (O(N) rather than O(N log N)) ---
    n = min(top_n, len(scores))
    if n == 0:
        return []

    # argpartition gives unsorted top-n; then sort just those n elements
    top_part = np.argpartition(scores, -n)[-n:]
    top_part = top_part[np.argsort(scores[top_part])[::-1]]

    # --- Build result dicts ---
    results = []
    for rank_idx in top_part:
        item_idx = int(indices[rank_idx])
        movie_id = int(movieids[item_idx])
        title, genres = movie_meta.get(movie_id, ("Unknown", "(no genres listed)"))
        results.append(
            {
                "movieId":         movie_id,
                "title":           title,
                "genres":          genres,
                "predicted_score": float(scores[rank_idx]),
            }
        )

    return results
