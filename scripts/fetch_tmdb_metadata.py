# Look up TMDB id for each movie in a user's raw ratings.csv export.
#
# Uses engine.match.resolve_tmdb_ids -- concurrent, rate-limited TMDB
# lookups (asyncio + httpx, shared with engine/content.py and
# scripts/fetch_tmdb_content.py) -- rather than a one-request-at-a-time
# synchronous loop with a fixed sleep between calls. Roughly an order of
# magnitude faster on a full Letterboxd export; see CHANGELOG.md.
#
# Usage:
#   python scripts/fetch_tmdb_metadata.py --user adeeb
#   python scripts/fetch_tmdb_metadata.py --user caroline-2026-09-09

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.match import resolve_tmdb_ids
from engine.paths import resolve_user_dir

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def main():
    parser = argparse.ArgumentParser(description="Look up TMDB ids for a user's raw Letterboxd ratings export.")
    parser.add_argument("--user", required=True, help="Short user name (e.g. 'adeeb') or exact data/ folder name")
    args = parser.parse_args()
    user_dir = resolve_user_dir(args.user)

    ratings = pd.read_csv(user_dir / "ratings.csv")
    ratings.drop(columns=["Letterboxd URI", "Date"], inplace=True)

    # (title, year, rating) tuples -- year cast to int (or None if missing),
    # matching resolve_tmdb_ids'/_search_one's Optional[int] expectation
    # rather than passing pandas' raw NaN-as-float through.
    ratings_tuples = [
        (row["Name"], int(row["Year"]) if pd.notna(row["Year"]) else None, row["Rating"])
        for _, row in ratings.iterrows()
    ]

    records, unmatched = asyncio.run(resolve_tmdb_ids(ratings_tuples, api_key=TMDB_API_KEY))

    tmdb_df = pd.DataFrame(records)
    tmdb_df.to_csv(user_dir / "tmdb_metadata.csv", index=False)
    print(f"Saved {len(tmdb_df)} rows to {user_dir / 'tmdb_metadata.csv'}.")
    if unmatched:
        print(f"\nNot found on TMDB, or no result within a plausible year ({len(unmatched)}):")
        for title in unmatched:
            print(f"  - {title}")


if __name__ == "__main__":
    main()
