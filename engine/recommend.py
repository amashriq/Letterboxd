# Recommendation-scoring utilities, spanning two generations of this project:
#
# CURRENT (used by scripts/predict.py, scripts/recommend.py, scripts/cv_alpha.py,
# scripts/cv_cascade.py): fold_in_user() + predict_rating() score against the
# PyTorch hybrid MF model (scripts/train_hybrid_mf.py) via a Ridge fold-in --
# see fold_in_user's own docstring for the full rationale.
#
# LEGACY (SVD-era, not called anywhere in the current pipeline): load_svd_artifacts(),
# _score_popularity(), _score_item_cf(), and recommend() implement a two-tier
# popularity/item-CF scorer against Funk SVD item vectors
# (scripts/archive/train_svd.py, now archived). Kept live intentionally
# (not archived) as a reference implementation of the earlier approach.
#
# load_movie_metadata() is current -- used by every script above to resolve a
# movieId to (title, genres).
#
# Public API:
#   meta = load_movie_metadata(Path("data/ml-32m/movies.csv"))
#   user_vector, user_bias = fold_in_user(matched, hybrid_artifacts)
#   rating = predict_rating(user_vector, user_bias, item_vector, item_bias, global_mean)

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
        Path to ``svd_item_vectors.npz`` produced by
        ``scripts/archive/train_svd.py`` (archived -- see this module's
        header comment). Expected keys: ``item_vectors`` (N×k), ``item_biases`` (N,),
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
    # Iterate over Series columns directly — yields Any, so int()/str() resolve
    # cleanly without the Scalar-type complaints from itertuples().
    meta: dict[int, tuple[str, str]] = {}
    for mid, title, genres in zip(df["movieId"], df["title"], df["genres"]):
        meta[int(mid)] = (str(title), str(genres))
    return meta


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
# Fold-in: estimate a user vector for the hybrid MF model (engine.content)
# ---------------------------------------------------------------------------

def fold_in_user(
    matched: list[dict],
    hybrid_artifacts: dict,
    alpha: float = 100.0,
) -> tuple[np.ndarray, float]:
    """
    Estimate a user vector for someone who was NOT part of
    ``scripts/train_hybrid_mf.py``'s training run -- i.e. everyone, since
    that script deliberately never bakes any specific user's ratings in.

    Fits a `Ridge regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_
    with the user's own known ratings as targets and the corresponding
    FROZEN, already-trained item vectors as features:

    .. math::

        \\text{rating}_i \\approx \\mu + b_i + b_u + \\mathbf{u} \\cdot \\mathbf{q}_i

    Only :math:`\\mathbf{u}` (``user_vector``) and :math:`b_u`
    (``user_bias``, Ridge's intercept) are unknowns -- everything else comes
    straight from ``hybrid_artifacts``. This is a few-millisecond closed-form
    fit, not a retrain: it's what makes a new/updated ``ratings.csv`` usable
    immediately instead of requiring hours of joint training.

    ``scripts/train_hybrid_mf.py`` trains the item vectors via EXPLICIT-rating
    MSE regression (not BPR/WARP pairwise ranking, despite the architecture
    being based on LightFM -- see that script's own header comment for why),
    so they're already on the 0.5-5 rating scale and Ridge here just
    calibrates a matching user vector against them, and
    :func:`predict_rating` returns a genuine rating estimate.

    Parameters
    ----------
    matched:
        Output of ``engine.match.match_ratings`` -- each dict must contain
        ``"movieId"`` (int) and ``"rating"`` (float).
    hybrid_artifacts:
        Dict returned by :func:`engine.content.load_hybrid_artifacts`.
    alpha:
        Ridge regularisation strength, passed straight to ``Ridge`` -- no
        ``n``-dependent scaling (an ``alpha/(alpha+n)`` adaptive version was
        tried and reverted; see git history around 2026-09-08 if reviving it).
        Higher = user vector pulled closer to zero (safer with few ratings);
        lower = fits the given ratings more tightly (only sensible with many
        of them).

        STATUS (as of 2026-09-09): default raised to ``100.0``, backed by a
        real leave-one-out cross-validation (``scripts/cv_alpha.py`` /
        ``scripts/cv_cascade.py``), not the one-off empirical case this
        docstring used to cite. Full sweep: latent-factor count k (10-256,
        each a separately fully-retrained model) x alpha (0.3-5000) x
        profile size n (20-286, subsampled with repeated draws from a
        286-rating profile to avoid confounding n with which person's
        ratings are harder to predict -- see the alpha-tuning memory).

        Two findings from that sweep: (1) k barely affects warm-item fold-in
        RMSE at all (~1-3% spread across the whole 10-256 range once each k
        gets its own best alpha) -- alpha is the dominant lever, not model
        size, so there's no k-dependent (or by extension k-per-n) formula to
        chase here. (2) After bracketing alpha until RMSE actually turned
        back up (not just stopping at an arbitrary upper bound -- the
        original ``10.0`` and even a first re-check up to 100 both undershot
        the real minimum), the true optimum for k=24 lands consistently in
        ~60-150 across 8 of 9 tested n, with ``100.0`` hitting or sitting
        inside that band at every one of them. One n=35 draw never turned up
        at all even out to alpha=5000 -- not a bug, just Ridge asymptoting
        toward a bias-only (zero-personalization) fit when a particular
        subsample has little exploitable signal; RMSE can't rise past that
        limit, only flatten.

        NEXT STEPS (planned, not started): an ``alpha/(alpha+n)`` adaptive
        scaling rule was tried once and reverted for being guessed rather
        than fit (see git history around 2026-09-08) -- the cascade sweep
        above is the fit-to-data version of that question, and its answer is
        "no strong n-dependence was found" rather than a formula. Remaining
        open threads: weight-decay (content-branch regularisation) was
        deliberately excluded from the sweep since it's invisible to
        warm-item RMSE -- testing it needs a genuinely different, cold-item
        evaluation; and n far outside the tested 20-286 range is unverified.

    Returns
    -------
    user_vector : np.ndarray, shape (k,)
    user_bias : float

    Raises
    ------
    ValueError
        If fewer than 2 of the matched ratings are on movies the model was
        actually trained on -- Ridge needs at least that many points to fit
        a k-dimensional vector plus an intercept meaningfully.
    """
    from sklearn.linear_model import Ridge

    global_mean = hybrid_artifacts["global_mean"]
    movieid_to_idx = hybrid_artifacts["movieid_to_idx"]
    item_embeddings = hybrid_artifacts["item_embeddings"]
    item_biases = hybrid_artifacts["item_biases"]

    X: list[np.ndarray] = []
    y: list[float] = []
    for m in matched:
        idx = movieid_to_idx.get(int(m["movieId"]))
        if idx is None:
            continue  # movie has no trained item vector -- can't use it to fold in
        X.append(item_embeddings[idx])
        y.append(float(m["rating"]) - global_mean - float(item_biases[idx]))

    if len(X) < 2:
        raise ValueError(
            f"Only {len(X)} matched rating(s) are on movies the model was "
            "trained on -- need at least 2 to fold in a user vector."
        )

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(np.array(X), np.array(y))
    return ridge.coef_, float(ridge.intercept_)


def predict_rating(
    user_vector: np.ndarray,
    user_bias: float,
    item_vector: np.ndarray,
    item_bias: float,
    global_mean: float,
) -> float:
    """
    Predicted rating for one (user, item) pair, on the same scale as the
    training ratings (MovieLens: 0.5-5.0 in 0.5 steps -- clamp/round at the
    call site if you need a display value on that exact scale).

    ``item_vector``/``item_bias`` should come from
    :func:`engine.content.get_item_representation`, which already handles
    the warm/cold split.
    """
    return global_mean + user_bias + item_bias + float(np.dot(user_vector, item_vector))


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

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
