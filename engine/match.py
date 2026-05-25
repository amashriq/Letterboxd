# Async TMDB → MovieLens matching module for the recommendation engine.
#
# Given a list of (title, year, rating) tuples from a Letterboxd CSV export,
# concurrently searches the TMDB search/movie endpoint for each film and
# cross-references results against the MovieLens links.csv bridge to produce
# MovieLens-matched records ready for scoring.
#
# Public API:
#   load_tmdb_to_ml(links_path)           → dict[int, int]
#   match_ratings(ratings, tmdb_to_ml)    → (matched, unmatched)

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

    Tries ``primary_release_year`` first (more precise); falls back to a bare
    title search if that returns no results.  Returns a metadata dict on
    success, ``None`` if nothing is found.

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

    # Year-scoped search first; skip if year is unknown
    results: list = []
    if year is not None:
        results = await fetch({**base_params, "primary_release_year": year})

    if not results:
        results = await fetch(base_params)  # fallback: bare title, no year filter

    if not results:
        return None

    best = results[0]
    return {
        "letterboxd_title": title,
        "tmdb_id":          best["id"],
        "original_title":   best.get("original_title"),
        "release_year":     (best.get("release_date") or "")[:4] or None,
        "rating":           rating,
    }


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
    Concurrently look up each ``(title, year, rating)`` on TMDB and
    cross-reference against MovieLens ``links.csv`` to find a ``movieId``.

    Uses a sliding-window :class:`RateLimiter` (default: 40 req / 10 s) and
    an ``asyncio.Semaphore`` to cap simultaneous in-flight requests.
    All lookups run concurrently via
    `asyncio.gather <https://docs.python.org/3/library/asyncio-task.html#asyncio.gather>`_.

    Parameters
    ----------
    ratings:
        List of ``(title, year, rating)`` tuples parsed from a Letterboxd
        ``ratings.csv`` export.  ``year`` may be ``None``.
    tmdb_to_ml:
        ``tmdbId → movieId`` mapping returned by :func:`load_tmdb_to_ml`.
        Pass the same object on every call — it is read-only.
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

    matched:   list[dict] = []
    unmatched: list[str]  = []

    for (title, _, _), result in zip(ratings, results):
        if result is None:
            unmatched.append(title)
            continue
        movie_id = tmdb_to_ml.get(result["tmdb_id"])
        if movie_id is None:
            unmatched.append(title)
        else:
            matched.append({"movieId": movie_id, **result})

    return matched, unmatched
