import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE = "https://api.themoviedb.org/3"

data_path = Path('data')


def lookup_tmdb(name, year, rating, api_key):
    def search(params):
        resp = requests.get(f"{TMDB_BASE}/search/movie", params=params)
        resp.raise_for_status()
        return resp.json().get('results', [])

    # Search with release year first, then fallback to without release year if no results
    results = search({'api_key': api_key, 'query': name, 'primary_release_year': year, 'include_adult': False})
    if not results:
        results = search({'api_key': api_key, 'query': name, 'include_adult': False})

    # Return None if no results, otherwise take first result
    if not results:
        return None
    best = results[0]
    
    return {
        'letterboxd_title': name,
        'tmdb_id': best['id'],
        'original_title': best.get('original_title'),
        'release_year': best.get('release_date', '')[:4] or None,
        'rating': rating,
    }


ratings = pd.read_csv(data_path / 'ratings.csv')
ratings.drop(columns=['Letterboxd URI', 'Date'], inplace=True)

records, unmatched = [], []
for _, row in ratings.iterrows():
    result = lookup_tmdb(row['Name'], row['Year'], row['Rating'], TMDB_API_KEY)
    if result:
        records.append(result)
    else:
        unmatched.append(row['Name'])
    time.sleep(0.05)

tmdb_df = pd.DataFrame(records)
tmdb_df.to_csv(data_path / 'tmdb_metadata.csv', index=False)
print(f"Saved {len(tmdb_df)} rows.")
if unmatched:
    print(f"\nNot found on TMDB ({len(unmatched)}):")
    for title in unmatched:
        print(f"  - {title}")
