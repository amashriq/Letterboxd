# Find which movies in tmdb_metadata.csv are also in the MovieLens dataset,
# and save the matched and unmatched sets separately for downstream use.

import pandas as pd
from pathlib import Path

data_path = Path('data')

tmdb_df = pd.read_csv(data_path / 'tmdb_metadata.csv')
links = pd.read_csv(data_path / 'ml-32m' / 'links.csv')

matched = tmdb_df.merge(links, left_on='tmdb_id', right_on='tmdbId', how='inner').drop(columns=['tmdbId', 'imdbId'])
unmatched = tmdb_df[~tmdb_df['tmdb_id'].isin(matched['tmdb_id'])]

matched.to_csv(data_path / 'movielens_matched.csv', index=False)
unmatched.to_csv(data_path / 'movielens_unmatched.csv', index=False)

print(f"Your ratings:      {len(tmdb_df)}")
print(f"Matched in ml-32m: {len(matched)}")
print(f"Not found:         {len(unmatched)}")
