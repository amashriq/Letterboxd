# Benchmark: the old synchronous, one-request-at-a-time TMDB matching loop
# (reconstructed here exactly as it worked before the engine.match.
# resolve_tmdb_ids rewrite -- see CHANGELOG.md's 2026-09-09 part 2, item 2)
# vs. the current async, rate-limited, concurrent version, on the same list
# of titles. Produces the real numbers behind the "roughly an order of
# magnitude faster" claim in README.md, instead of leaving it as an
# architectural argument with no measurement.
#
# Uses a fixed list of well-known public movie titles (not personal ratings
# data) so the benchmark is reproducible by anyone with a TMDB_API_KEY.
#
# Usage:
#   python scripts/bench_tmdb_match.py

import asyncio
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.match import resolve_tmdb_ids

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

# 40 real titles spanning eras/genres -- large enough to average out
# per-request latency noise, small enough to run in well under a minute
# either way.
BENCH_TITLES: list[tuple[str, int]] = [
    ("The Godfather", 1972), ("Pulp Fiction", 1994), ("The Matrix", 1999),
    ("Spirited Away", 2001), ("Parasite", 2019), ("Mad Max: Fury Road", 2015),
    ("Inception", 2010), ("The Dark Knight", 2008), ("Whiplash", 2014),
    ("Get Out", 2017), ("La La Land", 2016), ("Arrival", 2016),
    ("Everything Everywhere All at Once", 2022), ("Oppenheimer", 2023),
    ("Dune", 2021), ("Interstellar", 2014), ("Knives Out", 2019),
    ("The Grand Budapest Hotel", 2014), ("Moonlight", 2016), ("Her", 2013),
    ("Spider-Man: Into the Spider-Verse", 2018), ("Portrait of a Lady on Fire", 2019),
    ("Blade Runner 2049", 2017), ("The Social Network", 2010), ("Nomadland", 2020),
    ("Free Solo", 2018), ("The Grand Tour", 1953), ("Amelie", 2001),
    ("City of God", 2002), ("No Country for Old Men", 2007), ("There Will Be Blood", 2007),
    ("Children of Men", 2006), ("Eternal Sunshine of the Spotless Mind", 2004),
    ("The Shawshank Redemption", 1994), ("Fight Club", 1999), ("Goodfellas", 1990),
    ("Se7en", 1995), ("The Prestige", 2006), ("Memento", 2000), ("Coco", 2017),
]


def lookup_tmdb_sync(name: str, year: int, api_key: str) -> None:
    """Old approach: two blocking requests.get calls per title, sequential."""
    def search(params):
        resp = requests.get(f"{TMDB_BASE}/search/movie", params=params)
        resp.raise_for_status()
        return resp.json().get("results", [])

    search({"api_key": api_key, "query": name, "primary_release_year": year, "include_adult": False})
    search({"api_key": api_key, "query": name, "include_adult": False})


def bench_sync(titles: list[tuple[str, int]], api_key: str) -> float:
    t0 = time.perf_counter()
    for name, year in titles:
        lookup_tmdb_sync(name, year, api_key)
        time.sleep(0.05)  # matches the old fetch_tmdb_metadata.py's per-title sleep
    return time.perf_counter() - t0


def bench_async(titles: list[tuple[str, int]], api_key: str) -> float:
    ratings = [(name, year, 0.0) for name, year in titles]  # rating unused by resolve_tmdb_ids's TMDB call itself
    t0 = time.perf_counter()
    asyncio.run(resolve_tmdb_ids(ratings, api_key=api_key))
    return time.perf_counter() - t0


def main():
    if not TMDB_API_KEY:
        sys.exit("TMDB_API_KEY not set -- required for this benchmark.")

    print(f"Benchmarking {len(BENCH_TITLES)} titles...\n")

    print("Running OLD synchronous, one-request-at-a-time version...")
    sync_time = bench_sync(BENCH_TITLES, TMDB_API_KEY)
    print(f"  {sync_time:.2f}s\n")

    print("Running NEW async, rate-limited, concurrent version...")
    async_time = bench_async(BENCH_TITLES, TMDB_API_KEY)
    print(f"  {async_time:.2f}s\n")

    speedup = sync_time / async_time if async_time > 0 else float("inf")
    print(f"=== Result: {speedup:.1f}x faster ({sync_time:.2f}s -> {async_time:.2f}s, {len(BENCH_TITLES)} titles) ===")


if __name__ == "__main__":
    main()
