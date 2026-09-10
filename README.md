# Letterbox

A personal movie-recommendation pipeline that turns a raw
[Letterboxd](https://letterboxd.com/) ratings export into predicted star
ratings and top-N recommendations, using a hybrid matrix-factorization model
trained on [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) and
content metadata from [TMDB](https://www.themoviedb.org/).

## The problem

I use [Letterboxd](https://letterboxd.com/) as my main movie-review app, and
noticed that its own recommendation system — built from your own reviews —
is paywalled. That looked like a great opportunity to try using machine
learning to build my own recommendation model instead.

I watch a lot of new movies, since I love going to the theater, so part of
the challenge was fixing the [cold-start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)):
the largest public ratings dataset available to train on,
[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/), only has
ratings collected through October 2023, so anything released more recently
has zero rating history for a collaborative-filtering model to learn from —
exactly the kind of movie I most want a recommendation for. It's really a
two-sided version of the same problem: **cold items** (a movie with no
MovieLens history at all, new release or otherwise) and **cold users** (my
own ratings, or anyone else's, are never part of training either — see
Approach, below — so every person scoring their own Letterboxd export is
effectively cold too).

## Approach

**Hybrid hybrid, in two senses.** The model itself blends collaborative
(identity embeddings, learned per MovieLens movie) and content-based signal
(TMDB genres/cast/director/keywords) so every movie is scorable even without
rating history. Separately, *scoring a specific person* blends a one-time
expensive model train with a cheap per-user closed-form fit ("fold-in") —
see below.

### 1. Model: hybrid matrix factorization (PyTorch)

[`scripts/train_hybrid_mf.py`](scripts/train_hybrid_mf.py) trains a
[matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
model on all 32M MovieLens ratings, with an architecture based on
[LightFM](https://arxiv.org/abs/1507.08439) (reimplemented directly in
[PyTorch](https://pytorch.org/docs/stable/index.html), not the `lightfm`
package itself):

```
score(user, item) = user_vec · (item_vec + content_row @ content_embeddings)
                   + user_bias + item_bias + content_row @ content_biases
```

Every movie's score is the sum of a learned **identity** term (its own
trained embedding, like a classic MF model) and a **content** term (its TMDB
genres/keywords/director/cast, projected through a shared feature-embedding
matrix). A movie the model has never seen rated — brand new, or just below
the 20-rating floor to get its own identity embedding — still gets a real
score from the content term alone.

Two design choices worth calling out:
- **Trained on explicit ratings (MSE against real 0.5–5★ values), not
  implicit-feedback ranking** (BPR/WARP, what the `lightfm` library itself
  optimizes for — see
  [Rendle et al., 2009](https://arxiv.org/abs/1205.2618)). BPR only learns
  relative order, not a rating on any particular scale; since the actual
  product here is a predicted star rating, training has to target that scale
  directly.
- **Sparse gradient tables** (`nn.Embedding(..., sparse=True)` +
  [`SparseAdam`](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html))
  so a training step costs `O(batch size)`, not `O(n_users + n_items)` —
  necessary at 200k+ users and 20k+ items.

### 2. Content features (TMDB)

[`scripts/fetch_tmdb_content.py`](scripts/fetch_tmdb_content.py)
asynchronously fetches genres, cast, director, keywords, runtime, and
vote/budget/revenue stats for every MovieLens film from the
[TMDB API](https://developer.themoviedb.org/reference/intro/getting-started),
then builds a sparse feature matrix: genre one-hot, a handful of normalized
numerics (including a
[Bayesian-averaged](https://www.fxsolver.com/browse/formulas/Bayesian+average)
vote score to avoid trusting a 9.5/10 from 3 voters), keyword
[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
and billing-weighted director/cast binaries.

### 3. Fold-in: scoring a specific person without retraining

Training never sees anyone's personal ratings — the model is trained purely
on the anonymous MovieLens population, so it stays valid no matter whose
Letterboxd export you score next. A person's own taste is captured at
*inference* time via **fold-in**: a
[Ridge regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
(`engine/recommend.py:fold_in_user`) fit against that person's known ratings,
using the frozen, already-trained item vectors as features. It's a
few-millisecond closed-form fit, not a retrain — a new or updated
`ratings.csv` is usable immediately.

### 4. Matching: Letterboxd → TMDB → MovieLens

A Letterboxd export only has titles, years, and star ratings — no id that
links to either TMDB or MovieLens. `scripts/fetch_tmdb_metadata.py` resolves
each title to a TMDB id (picking the closest release-year match among
candidates, not blindly trusting TMDB's top search hit — real title
collisions like *Mean Girls* (2004 vs. 2024) or *The Lion King* (1994 vs.
2019) make that unsafe), then `scripts/match_movielens.py` cross-references
those TMDB ids against MovieLens's `links.csv` to get a `movieId`.

### Pipeline

```
Letterboxd export (data/<user>-<date>/ratings.csv)
        │  scripts/fetch_tmdb_metadata.py --user <name>
        ▼
tmdb_metadata.csv  (+ TMDB id per title)
        │  scripts/match_movielens.py --user <name>
        ▼
movielens_matched.csv  (+ MovieLens movieId per title)
        │
        │   ┌─ ml-32m/ratings.csv ──► scripts/train_hybrid_mf.py ──► hybrid_mf_artifacts.npz
        │   └─ TMDB per-film data  ──► scripts/fetch_tmdb_content.py ──► tmdb_content_*.npz
        ▼
scripts/predict.py / scripts/recommend.py   (fold in this person, score movies)
```

`scripts/cv_alpha.py` / `scripts/cv_cascade.py` are the evaluation layer —
[leave-one-out cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)
of the fold-in step's Ridge `alpha`, used to pick that hyperparameter from
held-out error instead of guessing (see Results, below).

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

`data/` is gitignored (dataset + personal exports + trained model weights —
several GB, none of it belongs in git), so a fresh clone needs it populated
before anything runs:

1. Download [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)
   and unzip it to `data/ml-32m/`.
2. Get a [TMDB API key](https://developer.themoviedb.org/reference/intro/getting-started)
   and put it in a `.env` file at the repo root: `TMDB_API_KEY=...`.
3. Export your ratings from Letterboxd (Settings → Import & Export → Export)
   and drop `ratings.csv` into `data/<name>-<YYYY-MM-DD>/` (e.g.
   `data/alice-2026-09-08/`).

## Running the pipeline

```bash
# 1. Per-user: Letterboxd → TMDB → MovieLens
python scripts/fetch_tmdb_metadata.py --user alice
python scripts/match_movielens.py --user alice

# 2. Shared, one-time (or whenever MovieLens/TMDB data changes): content
#    features + model training. fetch_tmdb_content.py is a full pass over
#    ~87k films (~5 hrs); train_hybrid_mf.py is a full pass over 32M ratings
#    (~11 hrs on CPU) — both support --limit/--dry-run for a quick smoke test.
python scripts/fetch_tmdb_content.py --limit 50    # or the full run, no --limit
python scripts/train_hybrid_mf.py --dry-run        # or the full run, no --dry-run

# 3. Score: fold this person's ratings in and predict/recommend
python scripts/predict.py --user alice --title "Mad Max: Fury Road" --year 2015
python scripts/recommend.py --user alice --genre Romance,Comedy --top-n 5

# 4. Evaluate the fold-in hyperparameter (alpha) by leave-one-out CV
python scripts/cv_alpha.py --user alice
```

No Letterboxd export or `TMDB_API_KEY`? `jupyter notebook notebooks/demo.ipynb` runs entirely offline against the trained model with a synthetic taste profile. `pytest` runs the unit tests (also no data files or network required).

## Results

### Fold-in (personalization)

The number a training run prints each epoch (`compute_rmse` in
`train_hybrid_mf.py`) is a **training-fit signal, not a generalization
estimate** — it's sampled from the same interactions the model was just
fit on. The numbers below are genuinely held-out instead: each is computed
by holding out one rating at a time, re-fitting the fold-in Ridge regression
on the rest, and scoring the held-out one.

| Evaluation | Profile size | RMSE (0.5–5★ scale) |
|---|---|---|
| Leave-one-out, single profile, **deployed default** (`data/<user>-<date>/alpha_cv_results.csv`) | 52 warm ratings | **0.81** at `alpha=100` (the value `fold_in_user` actually ships with), vs. **1.06** at scikit-learn's literal default (`alpha=1.0`) — a ~24% RMSE reduction from tuning alone |
| Leave-one-out, single profile, **best tested alpha** (same CSV) | 52 warm ratings | **0.79** at `alpha=30` — lower than the deployed value, but `alpha=100` was chosen for robustness across profile sizes (see below), not because 30 was untested |
| Leave-one-out, cascade sweep across profile sizes (`data/cascade-<date>/cascade_summary.csv`) | 20–286 warm ratings, second profile | **0.90–0.99** best-case per profile size — each point is its own (`k`, `alpha`) winner from the full grid, not one fixed configuration evaluated across all of them |

The ~24% gap (`python scripts/cv_alpha.py --user alice --alphas
1,30,100,1000,100000` — same leave-one-out harness, same 52-rating profile,
same deployed `hybrid_mf_artifacts.npz`) is against `Ridge`'s literal
out-of-the-box default, i.e. what fold-in would have scored before any
tuning happened at all, compared against what it actually ships with today
— not a synthetic upper bound and not the single best point in a sweep.
(The regularization curve isn't monotonic across this whole range — RMSE
bottoms out around `alpha=30` for this specific 52-rating profile, then
rises again through `alpha=100` and `alpha=100000`; `alpha=100` still beats
the untuned default, it just isn't the single lowest point *for this one
profile*, which is exactly why the cascade sweep — checking whether the
best alpha holds across many profile sizes, not just one — mattered.)

That cascade sweep (latent-factor count `k` × fold-in `alpha` × profile size
`n`, each `k` a fully separate retrained model) is also what the current
`fold_in_user` default (`alpha=100.0`) is based on — see its docstring in
[`engine/recommend.py`](engine/recommend.py) for the full analysis, including
the one counterintuitive finding: latent-factor count barely moved warm-item
RMSE at all (~1–3% spread across `k=10..256`) once each `k` got its own
best-fit `alpha` — regularization strength turned out to matter far more than
model size for this particular fold-in step.

### Core model — held-out RMSE + content-branch ablation

`hybrid_mf_artifacts.npz` (the production model) was trained on 100% of
MovieLens's ratings with no held-out split — deliberately, since the whole
point of fold-in is scoring a new person without retraining the population
model at all (see `train_hybrid_mf.py`'s module docstring). That also means
there's no way to get a fair generalization number from it directly.
`scripts/eval_holdout.py` trains a separate model on a genuine 90/10
train/test split, otherwise matching production's configuration exactly —
all 200,948 users, 23,350 items, `k=64`, 30 epochs — then evaluates the
held-out 10% three ways from that one trained model:

| Variant | RMSE | MAE | What it measures |
|---|---|---|---|
| `hybrid` | **0.82** | 0.61 | Full model — identity + content, exactly as trained |
| `identity_only` | 1.01 | 0.78 | Content term zeroed — plain biased MF, no TMDB signal at all |
| `content_only` | **0.86** | 0.66 | Identity term zeroed — the *exact* formula `engine/content.py:get_item_representation` uses to score a real cold item |

`content_only` isn't a hypothetical — it's production's actual cold-item
scoring formula, evaluated against held-out ratings on items the model has
real ground truth for. It lands within ~4.3% RMSE of the full warm model,
and outright beats `identity_only` — a stronger result for the cold-start
design than expected going in: the content branch is carrying most of the
predictive signal here, not just patching a gap collaborative filtering
leaves behind.

### Async TMDB matching

Benchmarked (`scripts/bench_tmdb_match.py`, 40 real titles, two runs):
**~2.0–2.1x faster** than the old synchronous, one-request-at-a-time loop.
Real TMDB round-trip latency dominates both loops, and the rate limiter
(tuned for TMDB's actual free-tier limit, not raw throughput) caps how much
concurrency buys on top of that.

## Repo layout

```
engine/          Shared library code: TMDB↔MovieLens matching, content
                 features, fold-in/scoring
scripts/         CLI entry points (one per pipeline stage — see Running,
                 above), cross-validation scripts, and evaluation/benchmark
                 scripts (eval_holdout.py, bench_tmdb_match.py)
notebooks/demo.ipynb
                 Offline walkthrough — fold in a synthetic taste profile,
                 see top-N recommendations, no network or personal data
                 required
tests/           Unit tests (pytest) for the trickiest pure-logic pieces —
                 TMDB year-matching, user-dir resolution, fold-in math
data/            Gitignored: MovieLens dataset, personal exports, trained
                 model artifacts (see data/README.md for what each file is
                 and which script produces it)
notebooks/archive/, scripts/archive/, archive/
                 Superseded exploratory notebooks and an earlier Funk-SVD
                 pipeline, kept for history rather than deleted
```
