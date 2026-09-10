# Find which movies in a user's tmdb_metadata.csv are also in the MovieLens
# dataset, and save the matched and unmatched sets separately for downstream use.
#
# Usage:
#   python scripts/match_movielens.py --user adeeb
#   python scripts/match_movielens.py --user caroline-2026-09-09

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.paths import DATA_DIR, resolve_user_dir


def main():
    parser = argparse.ArgumentParser(description="Match a user's TMDB-enriched ratings against MovieLens.")
    parser.add_argument("--user", required=True, help="Short user name (e.g. 'adeeb') or exact data/ folder name")
    args = parser.parse_args()
    user_dir = resolve_user_dir(args.user)

    tmdb_df = pd.read_csv(user_dir / 'tmdb_metadata.csv')
    links = pd.read_csv(DATA_DIR / 'ml-32m' / 'links.csv')

    matched = tmdb_df.merge(links, left_on='tmdb_id', right_on='tmdbId', how='inner').drop(columns=['tmdbId', 'imdbId'])
    unmatched = tmdb_df[~tmdb_df['tmdb_id'].isin(matched['tmdb_id'])]

    # ml-32m/links.csv has a handful of tmdbId values mapped to more than one
    # movieId (a known GroupLens/MovieLens community-links data-quality
    # artifact) -- an inner merge on one of those fans out into two rows for
    # one rating, silently double-counting it with an arbitrary movieId pick.
    # Route those to unmatched instead of guessing.
    ambiguous_mask = matched.duplicated('tmdb_id', keep=False)
    if ambiguous_mask.any():
        ambiguous = matched[ambiguous_mask]
        print(
            f"Dropped {ambiguous['tmdb_id'].nunique()} tmdb_id(s) with an ambiguous "
            f"movieId mapping in links.csv: {sorted(ambiguous['letterboxd_title'].unique())}"
        )
        matched = matched[~ambiguous_mask]
        unmatched = pd.concat([unmatched, ambiguous.drop(columns=['movieId'])], ignore_index=True)

    matched.to_csv(user_dir / 'movielens_matched.csv', index=False)
    unmatched.to_csv(user_dir / 'movielens_unmatched.csv', index=False)

    print(f"Your ratings:      {len(tmdb_df)}")
    print(f"Matched in ml-32m: {len(matched)}")
    print(f"Not found:         {len(unmatched)}")


if __name__ == "__main__":
    main()
