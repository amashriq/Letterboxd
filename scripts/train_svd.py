# Train Funk SVD on MovieLens 32M and save item vectors for fold-in and content branch targets.
# Run from the project root: python scripts/train_svd.py

import numpy as np
import pandas as pd
from pathlib import Path
from surprise import SVD, Dataset, Reader

data_path = Path('data')

N_FACTORS = 20   # k — keep < 35 (number of personal ratings for fold-in)
N_EPOCHS  = 40   # tuned via train/test RMSE analysis
LR        = 0.005
REG       = 0.1  # tuned via reg sweep on 500k sample

MIN_RATINGS = 20

# --- Load and filter ---
ratings = pd.read_csv(
    data_path / 'ml-32m' / 'ratings.csv',
    usecols=['userId', 'movieId', 'rating'],
)
counts = ratings['movieId'].value_counts()
valid_movies = counts[counts >= MIN_RATINGS].index
ratings = ratings[ratings['movieId'].isin(valid_movies)]
print(f"Movies with >= {MIN_RATINGS} ratings: {len(valid_movies):,}")
print(f"Ratings after filtering:              {len(ratings):,}")

# --- Train ---
reader   = Reader(rating_scale=(0.5, 5.0))
dataset  = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = dataset.build_full_trainset()

algo = SVD(n_factors=N_FACTORS, n_epochs=N_EPOCHS, lr_all=LR, reg_all=REG, verbose=True)
algo.fit(trainset)
print(f"\nItem matrix shape: {algo.qi.shape}")

# --- Check personal films are covered ---
matched = pd.read_csv(data_path / 'movielens_matched.csv')
trained_ids = set(int(trainset.to_raw_iid(i)) for i in range(trainset.n_items))
matched['in_trainset'] = matched['movieId'].isin(trained_ids)
print(f"Personal films in trainset: {matched['in_trainset'].sum()} / {len(matched)}")

# --- Save ---
movieids    = np.array([int(trainset.to_raw_iid(i)) for i in range(trainset.n_items)])
item_biases = algo.bi
global_mean = algo.trainset.global_mean

out_path = data_path / 'svd_item_vectors.npz'
np.savez(
    out_path,
    item_vectors=algo.qi,
    item_biases=item_biases,
    movieids=movieids,
    global_mean=np.array([global_mean]),
)
print(f"Saved {algo.qi.shape} item matrix to {out_path}")
