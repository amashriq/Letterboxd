# Unit tests for engine.match.pick_best_tmdb_match -- the year-proximity
# candidate matcher that exists specifically to avoid the real title
# collisions its own docstring cites (Mean Girls 2004/2024, The Lion King
# 1994/2019). This is the highest-value thing in the matching pipeline to
# pin down with a test: a regression here silently mismatches a movie
# rather than raising an error.

from engine.match import pick_best_tmdb_match


def test_no_candidates_returns_none():
    assert pick_best_tmdb_match([], year=2015) is None


def test_no_year_returns_first_candidate():
    candidates = [{"id": 1, "release_date": "2015-01-01"}, {"id": 2, "release_date": "2024-01-01"}]
    assert pick_best_tmdb_match(candidates, year=None) is candidates[0]


def test_picks_closest_year_not_first_result():
    # The Mean Girls 2004 vs. 2024 collision that motivated this function --
    # TMDB's own top search result isn't necessarily the right film.
    candidates = [
        {"id": 1, "release_date": "2024-01-16"},
        {"id": 2, "release_date": "2004-04-30"},
    ]
    best = pick_best_tmdb_match(candidates, year=2004)
    assert best["id"] == 2


def test_rejects_candidate_outside_tolerance():
    candidates = [{"id": 1, "release_date": "2019-01-01"}]
    assert pick_best_tmdb_match(candidates, year=2015) is None


def test_allows_one_year_tolerance_for_festival_vs_wide_release():
    candidates = [{"id": 1, "release_date": "2015-09-01"}]
    assert pick_best_tmdb_match(candidates, year=2014) is not None


def test_ties_keep_earlier_candidate():
    candidates = [{"id": 1, "release_date": "2015-01-01"}, {"id": 2, "release_date": "2015-06-01"}]
    best = pick_best_tmdb_match(candidates, year=2015)
    assert best["id"] == 1  # both are diff=0; first one found wins


def test_skips_candidates_with_unparseable_date():
    candidates = [{"id": 1, "release_date": ""}, {"id": 2, "release_date": "2015-01-01"}]
    best = pick_best_tmdb_match(candidates, year=2015)
    assert best["id"] == 2
