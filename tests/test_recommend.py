# Unit tests for engine.recommend's fold-in/scoring math -- small, synthetic
# artifacts so these run in milliseconds with no trained model or MovieLens
# data required.

import numpy as np
import pytest

from engine.recommend import fold_in_user, predict_rating


def _tiny_hybrid_artifacts() -> dict:
    """4 items, k=3 -- just enough for Ridge to fit a user vector + intercept."""
    item_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return {
        "global_mean": 3.5,
        "movieid_to_idx": {10: 0, 20: 1, 30: 2, 40: 3},
        "item_embeddings": item_embeddings,
        "item_biases": np.zeros(4),
    }


def test_fold_in_user_raises_below_minimum_ratings():
    artifacts = _tiny_hybrid_artifacts()
    with pytest.raises(ValueError):
        fold_in_user([{"movieId": 10, "rating": 5.0}], artifacts)


def test_fold_in_user_ignores_ratings_on_untrained_movies():
    artifacts = _tiny_hybrid_artifacts()
    # movieId 999 has no trained item vector -- silently dropped, so this
    # should still fail the >=2-warm-ratings check, not count it.
    with pytest.raises(ValueError):
        fold_in_user([{"movieId": 10, "rating": 5.0}, {"movieId": 999, "rating": 1.0}], artifacts)


def test_fold_in_user_returns_correct_shapes():
    artifacts = _tiny_hybrid_artifacts()
    matched = [{"movieId": 10, "rating": 5.0}, {"movieId": 20, "rating": 1.0}, {"movieId": 30, "rating": 3.0}]
    user_vector, user_bias = fold_in_user(matched, artifacts, alpha=1.0)
    assert user_vector.shape == (3,)
    assert isinstance(user_bias, float)


def test_predict_rating_is_dot_product_plus_biases():
    user_vector = np.array([1.0, 2.0, 0.0])
    item_vector = np.array([0.5, 0.5, 10.0])  # last dim shouldn't matter -- user_vector[2] is 0
    pred = predict_rating(user_vector, user_bias=0.1, item_vector=item_vector, item_bias=0.2, global_mean=3.5)
    expected = 3.5 + 0.1 + 0.2 + (1.0 * 0.5 + 2.0 * 0.5 + 0.0 * 10.0)
    assert pred == pytest.approx(expected)
