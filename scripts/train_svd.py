# Train Funk SVD on MovieLens 32M and save item vectors for fold-in and content branch targets.
# Run from the project root: python scripts/train_svd.py
#
# Set INCLUDE_PERSONAL_RATINGS = True to jointly train the personal user vector
# alongside all MovieLens users, rather than estimating it via fold-in post-hoc.
# The personal user vector and bias are saved to svd_user_vector.npz.

import numpy as np
import pandas as pd
from pathlib import Path
from surprise import SVD, Dataset, Reader

data_path = Path('data')

N_FACTORS = 10    # k — lowered from 20; better constrained by 35 personal ratings (ratio 35/11 ≈ 3.2x vs 1.7x at k=20)
N_EPOCHS  = 40    # tuned via train/test RMSE analysis
LR        = 0.005
REG       = 0.1   # tuned via reg sweep on 500k sample

MIN_RATINGS = 20

INCLUDE_PERSONAL_RATINGS = True   # jointly train personal user vector
PERSONAL_USER_ID         = 0      # userId=0 is not present in MovieLens 32M

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

# --- Optionally inject personal ratings ---
if INCLUDE_PERSONAL_RATINGS:
    matched = pd.read_csv(data_path / 'movielens_matched.csv')
    personal = matched[['movieId', 'rating']].copy()
    personal = personal[personal['movieId'].isin(valid_movies)]
    personal['userId'] = PERSONAL_USER_ID
    ratings = pd.concat([ratings, personal[['userId', 'movieId', 'rating']]], ignore_index=True)
    print(f"Injected {len(personal)} personal ratings as userId={PERSONAL_USER_ID}")

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

# --- Save item vectors ---
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

# --- Save personal user vector (joint training only) ---
if INCLUDE_PERSONAL_RATINGS:
    inner_uid   = trainset.to_inner_uid(PERSONAL_USER_ID)
    user_vector = algo.pu[inner_uid]   # shape (k,)
    user_bias   = algo.bu[inner_uid]   # scalar

    user_path = data_path / 'svd_user_vector.npz'
    np.savez(user_path, user_vector=user_vector, user_bias=np.array([user_bias]))
    print(f"Saved personal user vector (k={N_FACTORS}) and bias to {user_path}")
    print(f"User bias: {user_bias:.4f}  |  Vector norm: {np.linalg.norm(user_vector):.4f}")
