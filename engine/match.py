# Async TMDB → MovieLens matching module for the recommendation engine.
#
# Given a list of (title, year, rating) tuples from a Letterboxd CSV export,
# concurrently searches the TMDB search/movie endpoint for each film and
# cross-references results against the MovieLens links.csv bridge to produce
# MovieLens-matched records ready for scoring.
#
# Public API:
#   load_tmdb_to_ml(links_path)              → dict[int, int]
#   resolve_tmdb_ids(ratings)                → (resolved, unmatched) -- TMDB only, no MovieLens step
#   match_ratings(ratings, tmdb_to_ml)       → (matched, unmatched)   -- resolve_tmdb_ids + MovieLens cross-reference
#   pick_best_tmdb_match(candidates, year)   → best result dict, or None

import asyncio
import os
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# TMDB free-tier limits: https://developer.themoviedb.org/docs/rate-limiting
TMDB_BASE = "https://api.themoviedb.org/3"
_DEFAULT_RATE_LIMIT   = 40    # max requests per rate window
_DEFAULT_RATE_PERIOD  = 10.0  # rolling window size in seconds
_DEFAULT_MAX_CONCURRENT = 10  # simultaneous in-flight HTTP requests


class RateLimiter:
    """
    Sliding-window rate limiter backed by `asyncio.Semaphore`_.

    Acquiring a token marks the start of a request; the token is automatically
    returned ``period`` seconds later via ``loop.call_later``, ensuring at most
    ``rate`` requests in any rolling ``period``-second window.

    Both the concurrency semaphore and this limiter wrap every ``client.get``
    call individually, so fallback retries count against the same budget.

    .. _asyncio.Semaphore:
       https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore
    """

    def __init__(
        self,
        rate: int = _DEFAULT_RATE_LIMIT,
        period: float = _DEFAULT_RATE_PERIOD,
    ) -> None:
        self._sem = asyncio.Semaphore(rate)
        self._period = period

    async def __aenter__(self) -> "RateLimiter":
        await self._sem.acquire()
        return self

    async def __aexit__(self, *_) -> None:
        # call_later uses a synchronous callback — Semaphore.release() qualifies.
        loop = asyncio.get_running_loop()
        loop.call_later(self._period, self._sem.release)


def pick_best_tmdb_match(candidates: list[dict], year: Optional[int], tolerance: int = 1) -> Optional[dict]:
    """
    From a list of TMDB ``/search/movie`` result dicts (already deduped by
    ``id``, order preserved), return the one whose release year is within
    ``tolerance`` years of ``year`` (ties keep the earlier/higher-ranked
    candidate) -- ``tolerance=1`` to allow a genuine festival-vs-wide-release
    year discrepancy for the SAME film, not to paper over a different one.

    Returns ``None`` -- caller should treat the movie as unmatched -- if no
    candidate qualifies, even when farther-off candidates exist: TMDB's own
    ``primary_release_year`` search filter has been observed to NOT reliably
    exclude other years for real title collisions (e.g. Mean Girls 2004 vs.
    2024, The Lion King 1994 vs. 2019, Scary Movie 2000 vs. 2026) -- silently
    taking ``results[0]`` there produced clean swaps, not just imprecision.
    Falls back to the first candidate only if ``year`` is ``None`` (nothing
    to check against).
    """
    if not candidates:
        return None
    if year is None:
        return candidates[0]

    def year_of(c: dict) -> Optional[int]:
        y = (c.get("release_date") or "")[:4]
        return int(y) if y.isdigit() else None

    best, best_diff = None, None
    for c in candidates:
        cy = year_of(c)
        if cy is None:
            continue
        diff = abs(cy - year)
        if best_diff is None or diff < best_diff:
            best, best_diff = c, diff

    if best is None or best_diff > tolerance:
        return None
    return best


def load_tmdb_to_ml(links_path: "Path | str") -> dict[int, int]:
    """
    Load MovieLens ``links.csv`` and return a ``tmdbId → movieId`` lookup dict.

    Call once at startup and pass the result into :func:`match_ratings`.

    Parameters
    ----------
    links_path:
        Path to ``ml-32m/links.csv`` from the
        `MovieLens 32M dataset <https://grouplens.org/datasets/movielens/32m/>`_.

    Returns
    -------
    dict[int, int]
        Maps TMDB integer IDs to MovieLens integer movie IDs.
        Rows with a missing ``tmdbId`` are silently dropped.
    """
    df = pd.read_csv(links_path, usecols=["tmdbId", "movieId"])
    df = df.dropna(subset=["tmdbId"])
    df["tmdbId"] = df["tmdbId"].astype(int)
    return dict(zip(df["tmdbId"], df["movieId"]))


async def _search_one(
    client: httpx.AsyncClient,
    concurrency: asyncio.Semaphore,
    rate: RateLimiter,
    title: str,
    year: Optional[int],
    rating: float,
    api_key: str,
) -> Optional[dict]:
    """
    Search TMDB for a single movie.

    Queries both a year-scoped and a bare title search (not short-circuited
    on the first non-empty one -- TMDB's ``primary_release_year`` filter
    doesn't reliably exclude other years for real title collisions, so both
    result sets are pooled and the best match is picked by
    :func:`pick_best_tmdb_match`). Returns a metadata dict on success,
    ``None`` if nothing within a plausible year is found.

    `TMDB search/movie endpoint <https://developer.themoviedb.org/reference/search-movie>`_
    """
    base_params: dict = {
        "api_key": api_key,
        "query": title,
        "include_adult": False,
    }

    async def fetch(params: dict) -> list:
        async with concurrency:
            async with rate:
                try:
                    resp = await client.get(
                        f"{TMDB_BASE}/search/movie", params=params
                    )
                    resp.raise_for_status()
                    return resp.json().get("results", [])
                except httpx.HTTPError as exc:
                    # Log and continue — one failed lookup shouldn't abort the batch.
                    # If all lookups fail, suspect a bad API key.
                    print(f"  [warn] TMDB lookup failed for {title!r}: {exc}")
                    return []

    scoped = await fetch({**base_params, "primary_release_year": year}) if year is not None else []
    bare = await fetch(base_params)

    seen: dict[int, dict] = {}
    for c in scoped + bare:
        seen.setdefault(c["id"], c)

    best = pick_best_tmdb_match(list(seen.values()), year)
    if best is None:
        return None

    return {
        "letterboxd_title": title,
        "letterboxd_year":  year,
        "tmdb_id":          best["id"],
        "original_title":   best.get("original_title"),
        "release_year":     (best.get("release_date") or "")[:4] or None,
        "rating":           rating,
    }


async def resolve_tmdb_ids(
    ratings: list[tuple[str, Optional[int], float]],
    api_key: Optional[str] = None,
    rate_limit: int = _DEFAULT_RATE_LIMIT,
    rate_period: float = _DEFAULT_RATE_PERIOD,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    timeout: float = 15.0,
) -> tuple[list[dict], list[str]]:
    """
    Concurrently look up each ``(title, year, rating)`` on TMDB. This is the
    async fan-out core of this module -- a sliding-window :class:`RateLimiter`
    (default: 40 req / 10 s) plus an ``asyncio.Semaphore`` bounding
    simultaneous in-flight requests, with every lookup dispatched at once via
    `asyncio.gather <https://docs.python.org/3/library/asyncio-task.html#asyncio.gather>`_
    rather than one request at a time. Used directly by
    ``scripts/fetch_tmdb_metadata.py`` (which only needs a ``tmdb_id`` per
    title -- the MovieLens ``movieId`` cross-reference is
    ``scripts/match_movielens.py``'s separate pipeline stage), and wrapped by
    :func:`match_ratings` below for a caller that wants both stages fused
    into one call. ~2x faster on a full Letterboxd export than a synchronous,
    one-request-at-a-time loop with a fixed sleep between calls (what
    ``fetch_tmdb_metadata.py`` used before switching to this) --
    benchmarked in ``scripts/bench_tmdb_match.py``, run it yourself for a
    number specific to your network/TMDB response times. Real-world network
    latency per request (TMDB round-trip time) turns out to dominate over
    the artificial ``time.sleep(0.05)`` the old loop added, and this
    module's own rate limiter (40 req/10s, tuned for TMDB's free tier, not
    for raw throughput) caps how much concurrency actually buys here --
    both loops end up latency-bound, just with a shorter critical path on
    the async side.

    Parameters
    ----------
    ratings:
        List of ``(title, year, rating)`` tuples parsed from a Letterboxd
        ``ratings.csv`` export.  ``year`` may be ``None``.
    api_key:
        TMDB v3 API key.  Defaults to the ``TMDB_API_KEY`` environment variable.
    rate_limit:
        Max requests per ``rate_period`` seconds.  TMDB free tier: 40.
    rate_period:
        Rolling window size in seconds.
    max_concurrent:
        Max simultaneous in-flight HTTP connections.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    resolved : list[dict]
        One dict per title with a TMDB match, with keys: ``letterboxd_title``,
        ``letterboxd_year``, ``tmdb_id``, ``original_title``, ``release_year``,
        ``rating``.
    unmatched : list[str]
        Titles that couldn't be resolved -- not found on TMDB at all, or
        found but with no result within a plausible year (see
        :func:`pick_best_tmdb_match`).

    Raises
    ------
    ValueError
        If no API key is available.
    """
    key = api_key or os.getenv("TMDB_API_KEY")
    if not key:
        raise ValueError(
            "TMDB API key required — pass api_key= or set TMDB_API_KEY in .env."
        )

    limiter     = RateLimiter(rate_limit, rate_period)
    concurrency = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _search_one(client, concurrency, limiter, title, year, rating, key)
            for title, year, rating in ratings
        ]
        results = await asyncio.gather(*tasks)

    resolved:  list[dict] = []
    unmatched: list[str]  = []

    for (title, _, _), result in zip(ratings, results):
        if result is None:
            unmatched.append(title)
        else:
            resolved.append(result)

    return resolved, unmatched


async def match_ratings(
    ratings: list[tuple[str, Optional[int], float]],
    tmdb_to_ml: dict[int, int],
    api_key: Optional[str] = None,
    rate_limit: int = _DEFAULT_RATE_LIMIT,
    rate_period: float = _DEFAULT_RATE_PERIOD,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    timeout: float = 15.0,
) -> tuple[list[dict], list[str]]:
    """
    :func:`resolve_tmdb_ids`, plus a MovieLens ``links.csv`` cross-reference
    on top -- fuses ``fetch_tmdb_metadata.py``'s and ``match_movielens.py``'s
    pipeline stages into one call, for a caller that wants a ``movieId``
    directly instead of a two-step CSV round-trip. Same parameters as
    :func:`resolve_tmdb_ids`, plus:

    Parameters
    ----------
    tmdb_to_ml:
        ``tmdbId → movieId`` mapping returned by :func:`load_tmdb_to_ml`.
        Pass the same object on every call — it is read-only.

    Returns
    -------
    matched : list[dict]
        One dict per successfully matched film, with keys:
        ``movieId``, ``letterboxd_title``, ``tmdb_id``, ``original_title``,
        ``release_year``, ``rating``.
        These have a confirmed MovieLens entry and can be scored.
    unmatched : list[str]
        Titles that couldn't be resolved — either not found on TMDB at all,
        or found on TMDB but absent from MovieLens (typically recent releases).

    Raises
    ------
    ValueError
        If no API key is available.
    """
    resolved, unmatched = await resolve_tmdb_ids(
        ratings, api_key=api_key, rate_limit=rate_limit, rate_period=rate_period,
        max_concurrent=max_concurrent, timeout=timeout,
    )

    matched: list[dict] = []
    for result in resolved:
        movie_id = tmdb_to_ml.get(result["tmdb_id"])
        if movie_id is None:
            unmatched.append(result["letterboxd_title"])
        else:
            matched.append({"movieId": movie_id, **result})

    return matched, unmatched
