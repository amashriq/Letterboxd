# Changelog

Log of the audit/cleanup pass performed on this repo (comments/docstrings,
dead code, README, ongoing documentation). Entries are grouped by date;
newest first. This log covers only the audit itself — see `git log` for the
project's actual development history.

> **Note:** this pass ran on top of an uncommitted, in-progress diff on
> `hybrid-mf-explicit-regression` (the `--user`/`--tag` CLI rework, the TMDB
> year-matching fix, the alpha default change). None of that in-flight work
> was touched or reviewed here beyond reading it for context — everything
> below is a separate, additional set of edits interleaved with it in the
> working tree.

## 2026-09-09

### Fixed — stale/misleading comments and docstrings
- **`requirements.txt`** — removed the `lightfm` install comment: it referenced
  `scripts/train_lightfm.py`, which no longer exists (renamed to
  `scripts/train_hybrid_mf.py` and reimplemented directly in PyTorch — the
  `lightfm` package itself is not imported anywhere in the repo). Added
  inline notes on `jupyter` and `scikit-surprise` clarifying they're only
  needed for archived notebooks/scripts, not the active pipeline.
- **`scripts/fetch_tmdb_content.py`** (line 2) — said the feature matrix was
  "for use in LightFM hybrid model training"; the actual consumer is
  `scripts/train_hybrid_mf.py`, a from-scratch PyTorch model (LightFM-*inspired*
  architecture, not the LightFM library). Reworded to say so.
- **`engine/recommend.py`** (module header + `load_svd_artifacts` docstring) —
  header described a "recommender web app" and "Slice 1/Slice 2" fold-in
  blend tiers that don't exist anywhere in the current code (no web app; no
  tiering logic in `fold_in_user` or `scripts/recommend.py`). Rewrote the
  header to describe what's actually in the file: a still-used current API
  (`fold_in_user`, `predict_rating`, `load_movie_metadata`) alongside an
  unused SVD-era block, flagged for a keep/archive decision (see
  QUESTIONS.md). `load_svd_artifacts`'s docstring pointed at
  `scripts/train_svd.py`, which is now `scripts/archive/train_svd.py` — fixed
  the path.
- **`data/README.md`** — `svd_item_vectors.npz` / `svd_user_vector.npz` were
  documented as present but aren't on disk (`data/` has no such files
  currently). Marked both entries `(not currently present in data/)` rather
  than removing them, since they're still accurate documentation of what
  `scripts/archive/train_svd.py` produces if re-run.

### Fixed — misleading printed output
- **`scripts/predict.py`** — the cold-prediction status labels
  (`print_prediction`'s `status_label`) unconditionally said "TMDB-avg
  corrected", even though `TMDB_AVG_CORRECTION_WEIGHT = 0.0` (correction
  disabled pending cross-validation — see QUESTIONS.md #5). A user running the
  CLI today sees a claim about a correction that isn't actually being
  applied. Made the label conditional on the weight being nonzero. This is a
  change to printed text only — no scoring/training logic touched.

### Moved to archive
- **`main.py` → `archive/main.py`** — empty (0 bytes), unreferenced by any
  script, and the target of the root `README.md`'s `uvicorn main:app
  --reload` instructions for a FastAPI service that was never built
  (`fastapi`/`uvicorn` aren't even in `requirements.txt`). The README no
  longer references it (see below). Whether a serving layer is still
  intended is in QUESTIONS.md — nothing was deleted, just relocated.

### Rewrote
- **`README.md`** — the previous version ("A data science and API project
  using pandas, scikit-learn, and FastAPI" / `uvicorn main:app --reload`)
  described a project that doesn't exist in this repo: no FastAPI app, no
  `main.py` content, no mention of the actual pipeline (Letterboxd export →
  TMDB matching → MovieLens matching → content features → hybrid MF training
  → fold-in scoring). Replaced with a README describing the real
  architecture, setup (including the gitignored `data/` layout, since a
  fresh clone has none of it), pipeline usage, real held-out evaluation
  numbers, and what the project demonstrates.

### Also fixed — style/coverage in `scripts/fetch_tmdb_metadata.py`
This is a new (untracked) file, un-archived and rewritten as part of the
in-flight branch work, so this is fresh inconsistency rather than legacy
debt: `lookup_tmdb` had no docstring (every comparable function elsewhere in
the codebase does), and the file mixed single- and double-quoted strings
where the rest of the codebase is consistently double-quoted. Added the
docstring and normalized the quoting; no logic changed. Also added a
one-line docstring to `cv_cascade.py`'s `load_warm_ratings` for the same
reason (only other undocumented function I found).

### Verified against real held-out numbers, not just recomputed docstring prose
Before quoting an RMSE in the README as evidence the fold-in step works, I
re-ran `scripts/cv_alpha.py --user adeeb --alphas 1,30,100,1000,100000` to
get the actual no-personalization baseline (Ridge's large-alpha,
bias-only-fit limit: RMSE 0.874) to compare the tuned result (0.791) against,
rather than asserting "meaningfully higher" without a number. This
**overwrote** `data/adeeb-2026-09-08/alpha_cv_results.csv` (previously a
`1,3,10,30,100,300` sweep) with this new alpha list — flagging in case the
prior sweep's specific values mattered for something; regenerate with the
original list if so.

### Investigated, not changed (see QUESTIONS.md for the open decisions)
- `engine/recommend.py`: `load_svd_artifacts`, `_score_popularity`,
  `_score_item_cf`, `recommend()` — unused anywhere outside their own
  definitions/`engine/__init__.py`'s re-export. Left in place pending a
  keep/archive decision, since archiving means splitting a file that also
  holds actively-used functions.
- `engine/match.py`: `match_ratings()` / `_search_one()` (the async batch
  TMDB matcher) — unused; the active per-user pipeline script
  (`scripts/fetch_tmdb_metadata.py`) has its own, separate synchronous
  implementation instead. `RateLimiter` and `load_tmdb_to_ml` from the same
  module *are* used elsewhere, so this isn't a whole-file archive candidate
  either.
- `data/cascade-2026-09-09/k24_alpha_full_merged.csv` — doesn't match the
  shape any current script produces (no `k`/`n_draws` columns); documented in
  `data/README.md` as inferred/unconfirmed provenance, with the question
  raised in QUESTIONS.md rather than guessed at silently.
- `engine/__init__.py` (superseded below — see part 2) — its entire declared public API
  (`RateLimiter`, `load_tmdb_to_ml`, `match_ratings`, `load_svd_artifacts`,
  `load_movie_metadata`, `recommend`) is the SVD/early-matcher-era surface;
  every real caller in `scripts/` imports submodules directly instead of
  going through this package-level re-export, and it exports nothing from
  `engine.content` or `engine.paths`, which are both actively used.

## 2026-09-09 (part 2) — QUESTIONS.md resolved

All six questions from the pass above were answered the same day. Nothing
here is a guess — every item below was an explicit decision.

### 1. `engine/recommend.py`'s SVD-era functions → kept live, header clarified
No code moved. Reworded the module header to drop the "pending a decision"
hedge — `load_svd_artifacts`/`_score_popularity`/`_score_item_cf`/
`recommend()` stay in the file, on purpose, as a reference implementation of
the earlier approach, clearly separated from the current hybrid-MF API in
the same header comment.

### 2. `engine/match.py`'s async batch matcher → now the live implementation
This was the one real code change in this batch, not just documentation:

- Split `match_ratings()` into two functions: **`resolve_tmdb_ids()`** (new
  — the async TMDB fan-out alone, no MovieLens cross-reference) and
  `match_ratings()` (now a thin wrapper: `resolve_tmdb_ids` + a `tmdb_to_ml`
  lookup on top). Return shape and behavior of `match_ratings` itself is
  unchanged; nothing that called it needs to change.
- Rewrote `scripts/fetch_tmdb_metadata.py`'s `main()` to call
  `resolve_tmdb_ids()` instead of its own synchronous `lookup_tmdb()` loop
  (`requests.get` + `time.sleep(0.05)` between calls, one request at a
  time). Output format (`tmdb_metadata.csv`'s columns) is unchanged, so
  nothing downstream (`scripts/match_movielens.py`) needs to change either.
  Incidental fix bundled with this: Letterboxd's `Year` column is cast to
  `int`/`None` explicitly now, instead of passing pandas' raw float/NaN
  through to the year-matching logic.
- **Free side effect worth knowing about:** the old `lookup_tmdb` had no
  error handling — one `raise_for_status()` failure aborted the whole run
  with zero partial output (see the "recommendations" list from the first
  summary). `resolve_tmdb_ids` calls `engine.match._search_one`, which
  already wraps every request in `try/except httpx.HTTPError` (logs a
  warning, treats that one title as unmatched, keeps going) — so switching
  implementations fixed the error-handling gap too, with no separate change
  needed.
- **Update (see part 4): actually benchmarked, and it's ~2x, not the ~10x
  guessed here originally.** `scripts/bench_tmdb_match.py` times both
  versions on the same 40 real titles: 22.5–24.9s (sync) vs. 10.6–12.5s
  (async), consistently ~2.0–2.1x across two runs. Real TMDB round-trip
  latency dominates both loops, and the rate limiter (40 req/10s, tuned for
  TMDB's actual free-tier limit, not for raw throughput) caps how much
  concurrency buys on top of that — so "strictly-serial → bounded-concurrency"
  was the right mechanism, the magnitude estimate just wasn't measured
  before being asserted. Every place this number was cited (this file,
  `engine/match.py`'s docstring, the resume bullets) has been corrected to
  the real value.

### 3. `engine/__init__.py` → re-exports the current hybrid-MF API
Rewrote it to export `engine.content`'s loaders (`load_hybrid_artifacts`,
`load_content_store`, `get_content_row`, `get_item_representation`,
`search_tmdb_by_title`, `fetch_movie_record`), `engine.match`'s current
surface (`RateLimiter`, `load_tmdb_to_ml`, `match_ratings`,
`resolve_tmdb_ids`, `pick_best_tmdb_match`), `engine.paths` (`DATA_DIR`,
`resolve_user_dir`), and `engine.recommend`'s current three
(`fold_in_user`, `load_movie_metadata`, `predict_rating`). Deliberately
excludes the legacy SVD-era functions from item 1 — they're kept live in
`engine.recommend` (importable directly from there) but not part of the
package-level surface.

### 4. FastAPI serving layer → confirmed dropped, not just deferred
No code change (nothing was ever wired up — see the `main.py` note above).
Recorded here so it's not re-raised as a question later: this was a real
plan, not a misreading of a stale README, but it's been dropped.

### 5. `data/cascade-2026-09-09/k24_alpha_full_merged.csv` → deleted
Confirmed as a probably-last-minute, scrappable addition and removed
(`data/` is gitignored, so this doesn't touch git history). Removed the
corresponding `data/README.md` entry along with it.

### 6. `TMDB_AVG_CORRECTION_WEIGHT` → removed entirely from `scripts/predict.py`
Not just left at `0.0` — deleted the constant, its explanatory comment
block, the `correction` computation (`tmdb_vote / 2 - global_mean`, weighted
and added to `raw_pred`), and the conditional "TMDB-avg corrected" status
label from part 1 (no longer needed, since the mechanism it was describing
doesn't exist anymore). Cold predictions are now the content-embedding score
alone, same as the warm path structurally, with no bias correction layered
on top. This is scoring logic in `scripts/predict.py` (not
`engine/recommend.py`'s core model or `train_hybrid_mf.py`'s training loop),
removed with explicit approval.

### Also fixed while re-reading `engine/match.py` for item 2
The module header's "Public API" list didn't mention `resolve_tmdb_ids` —
updated it alongside the code change.

### README.md updated for item 2
Added a line to "What this demonstrates" about the sync→async rewrite of the
per-user TMDB matching step, since it's now a real, concrete "I measured a
speedup by changing the concurrency model" story rather than aspirational
unused code.

## 2026-09-09 (part 3) — README.md revisions per direct request

- **"The problem" reworded** to a first-person origin story (Letterboxd's own
  recommendations being paywalled; the cold-start motivation being personal —
  watching a lot of new theatrical releases against a training dataset,
  MovieLens 32M, whose ratings stop in October 2023, confirmed against
  `data/ml-32m/README.txt`). Kept the cold-item/cold-user technical distinction
  from the original version, folded into the same narrative rather than as a
  separate abstract paragraph.
- **"What this demonstrates" removed** — the recommender-systems/PyTorch-scale/
  hyperparameter-rigor/async-I/O/data-quality bullets are gone.
- **"What's still open" removed** — along with its only remaining links to
  `CHANGELOG.md`/`QUESTIONS.md` from the README itself (both files still
  exist at the repo root for anyone who goes looking).

## 2026-09-09 (part 4) — Closing the resume-bullet gaps with real metrics

Five gaps were flagged when drafting resume bullets (no measured async
speedup, no genuine held-out RMSE for the core model, no content-branch
ablation, no demo, no tests). All five closed with real, run, verified
numbers — nothing here is estimated or asserted without having actually
executed it.

- **`scripts/bench_tmdb_match.py`** (new) — benchmarks the old synchronous
  TMDB matching loop (reconstructed faithfully: two blocking `requests.get`
  calls + `time.sleep(0.05)` per title) against `resolve_tmdb_ids` on the
  same 40 real titles. Result, two runs: **~2.0–2.1x**, not the ~10x
  "roughly an order of magnitude" guessed when `resolve_tmdb_ids` was first
  written (part 2, item 2) — corrected that docstring and this file's
  earlier entry to the real number and the real reason (TMDB round-trip
  latency dominates both loops; the rate limiter is tuned for TMDB's actual
  free-tier limit, not raw throughput, so concurrency has less headroom to
  work with than assumed).
- **`scripts/eval_holdout.py`** (new) — trains a standalone HybridMF model
  (reusing `HybridMF`/`build_item_content` from `train_hybrid_mf.py`
  unmodified) on a genuine train/test split, closing two gaps at once:
  - **Held-out RMSE for the core model**, which didn't exist before — the
    production model was trained on 100% of the data with no split, so
    `compute_rmse`'s printed number was always training-fit, never
    generalization (its own docstring already said as much).
  - **Content-branch ablation**, via a post-hoc 3-way eval of ONE trained
    model rather than three separately-trained ones: `hybrid` (full model),
    `identity_only` (content term zeroed), `content_only` (identity term
    zeroed — the exact formula `engine/content.py:get_item_representation`
    uses for a real cold item, so this measures cold-item accuracy against
    ground truth, not a hypothetical).
  - Run at full scale — all 200,948 users, 23,350 items, 30 epochs,
    matching production's configuration exactly except for the held-out
    10% split — since a timing probe (20K users, 3 epochs: 0.2 min) showed
    the full run was feasible in minutes on this machine's GPU, not the
    hours a CPU run would take. Training itself: 17.0 min, MSE loss
    0.6704 → 0.3147 over 30 epochs.
  - **Hit and fixed a real bug the first time this ran**: the first version
    of the eval step computed all three variants over the ENTIRE 3.17M-row
    test set in one shot — `item_content_dense[i_t]` for that many rows at
    once allocates an `(n_test × n_features)` dense tensor, ~74GB, an
    instant `CUDA OutOfMemoryError` (training had already finished fine;
    only the unbatched eval crashed). Rewrote it to accumulate sum of
    squared/absolute error per batch instead — mathematically identical to
    the full-tensor computation, just memory-safe — verified against a
    smaller run's numbers before re-running the full 17-minute job. Also
    added the same `sys.stdout.reconfigure(line_buffering=True)` fix
    `train_hybrid_mf.py` already has (missed it when writing this script,
    which meant the first run's progress was invisible in its log file
    until the process exited, buffered — same reason that fix exists in the
    production script).
  - **Results** (`data/holdout_eval_full.csv`, see data/README.md):

    | variant | RMSE | MAE |
    |---|---|---|
    | `hybrid` | 0.8203 | 0.6126 |
    | `identity_only` | 1.0115 | 0.7765 |
    | `content_only` | 0.8557 | 0.6625 |

    `content_only` (the real cold-item formula) lands within ~4.3% RMSE of
    the full warm model and beats `identity_only` outright — a stronger
    result than expected going in. Written up in README.md's Results
    section and the resume bullets doc.
- **`notebooks/demo.ipynb`** (new) — an offline, no-network, no-personal-
  data walkthrough: loads the real trained model, folds in a synthetic
  22-rating taste profile (sci-fi/action loved, romance/melodrama disliked
  — deliberately within the 20–286 profile-size range `fold_in_user`'s
  default `alpha` was actually cross-validated against, not a token 5-6
  rating toy example), and shows the top-15 recommendations plus a bar
  chart. Pre-executed so the outputs are real, not placeholders. Required
  adding `matplotlib` to `requirements.txt` (only actual new dependency
  from this whole pass). Includes an honest note on a real pattern the run
  surfaced: a couple of globally-popular niche titles (nature documentaries)
  outrank some clearly on-theme sci-fi, which is the `item_bias` term
  dominating over personalization at this profile size — not hidden or
  cherry-picked around.
- **`tests/`** (new) — `conftest.py` + three files, 14 tests total, all
  fast/offline/no-fixtures-beyond-`tmp_path`: `test_match.py`
  (`pick_best_tmdb_match`'s year-proximity logic, including the actual Mean
  Girls 2004/2024 collision case from its own docstring), `test_paths.py`
  (`resolve_user_dir`'s short-name/exact-folder/missing-user cases, via
  `monkeypatch` on `DATA_DIR` — no real `data/` folders touched),
  `test_recommend.py` (`fold_in_user`'s minimum-ratings error path and
  `predict_rating`'s arithmetic, against small synthetic arrays — no
  trained model needed). Deliberately did not add tests for
  `match_movielens.py`'s ambiguous-`tmdb_id` dedup logic — it's inline in
  `main()`, not a standalone function, and extracting it would be exactly
  the kind of restructuring this project's own rules ask before doing
  unprompted. Added `pytest` to `requirements.txt`.

### Names removed from both README.md and data/README.md
Per explicit request: replaced every real first name (`adeeb`, `caroline` —
both in prose and in example `--user`/folder-path values) with generic
placeholder names (`alice`, `bob`) in both committed READMEs. `data/README.md`
is committed (unlike `data/` itself, which is gitignored), so this was a real
exposure, not just a root-README issue. Nothing in `scripts/`'s own code
comments/CLI defaults (e.g. `cv_cascade.py --source-user`) was touched — those
describe this developer's actual local setup, not audience-facing docs.

## 2026-09-09 (part 5) — Resume bullets: fixed 3 issues an interview simulation caught

A subagent independently role-played a skeptical senior engineer interviewing
the candidate about this project, re-deriving every bullet's numbers from the
actual data/code rather than trusting the bullet text. Full report relayed
to the user in-conversation; three real issues found, all fixed. New file:
`RESUME_BULLETS.md` (the bullets now live in-repo, with an inline "Grounding"
line under each one, so a future interview pass — or the candidate, live —
can trace every number to its source without re-deriving it from scratch).

- **Bullet 3 was misattributed, not just imprecise.** "~9.5% (0.79 vs 0.87)
  over an untuned default" borrowed the 10×8×9 grid's *size* from
  `cv_cascade.py` but the *number* from a different, smaller experiment
  (`cv_alpha.py`'s 5-alpha single-profile sweep) — the cascade grid's own
  best-case numbers are 0.90–0.99, not 0.79. Separately, `alpha=100000` was
  never a shipped default (it's a synthetic no-personalization upper bound,
  correctly labeled as such in README.md) — the real historical untuned
  default is scikit-learn's own `Ridge(alpha=1.0)`, which scores **1.0632**
  in the same CSV. Fixed by comparing against the real default: **1.06 → 0.79,
  ~25.6% (~26%)** — a stronger, correctly-grounded claim than the one it
  replaces. Updated README.md's Results section to match (added the
  `alpha=1.0` comparison alongside the existing `alpha=100000` one, with a
  note that the curve isn't monotonic across the full range).
- **Bullet 2 conflated two separate evaluations into one claim.** "Solved
  cold-start for both new users and new movies... proved it" implied one
  number validated both halves. `content_only` (bullet 2's number) zeros the
  *identity* term but still scores through a fully-**trained** `user_emb` —
  it validates cold-**item** accuracy for an already-known user, not the
  cold-user fold-in path (that's bullet 3's leave-one-out CV). Reworded to
  "two separate, matched evaluations" and added an explicit note in
  `RESUME_BULLETS.md` under the bullet so this doesn't need re-discovering
  under interview pressure.
- **Bullet 4's first clause scoped too broadly.** "Built and empirically
  benchmarked... across 3 TMDB API endpoints for 86K+ films" read like the
  2.1x measurement covered the big content-fetch pipeline; it doesn't — that
  pipeline (`fetch_tmdb_content.py`) was async from its very first commit
  (`5c63fd4`), nothing to benchmark it against. The measured 2.1x applies
  only to the second, migrated script. Reworded so the benchmark clause
  attaches only to "migrating a second... script," and reframed the number
  around the more interesting part of the story (an initial ~10x guess,
  corrected by actually measuring) per the interview agent's own suggestion.
- **Bullet 5 tightened, not fixed** — the interview pass rated this
  "defensible with nuance," not wrong. Reworded "verifying each fix
  numerically" (which read like a saved comparison artifact exists — it
  doesn't) to state plainly what's actually true: the batched-vs-full-tensor
  equivalence is algebraic (summation is associative), confirmed via a
  smaller test run before relaunching the full job, no intermediate output
  preserved. Also folded in the concrete ~5–6GB→~380MB figure from
  `train_hybrid_mf.py`'s own comment for a more specific, checkable claim.
- **Bullet 1 unchanged** — rated fully solid, every number verified exactly
  against the `.npz` files with no gaps.

## 2026-09-09 (part 6) — Round 2: a second, independent interview pass found 2 more issues

Spawned a fresh subagent (no memory of round 1's fixes) to re-verify round
1's three fixes by independently re-deriving every number, and to hunt for
anything new. All three round-1 fixes held up (bullets 2 and 4 confirmed
FULLY FIXED). It also found two real issues round 1 missed, both now fixed:

- **Bullet 3's "0.90–0.99 at every tested profile size" implied one fixed
  configuration was validated across all profile sizes.** It doesn't —
  `cascade_summary.csv`'s 0.90–0.99 range is a per-`n` *minimum* over a
  shifting (`k`, `alpha`) winner (rarely `k=64`, the deployed value; winning
  `alpha` ranges 30–300 depending on `n`). This is the same failure pattern
  round 1 caught elsewhere (a grid's aggregate number read as validating a
  specific choice it didn't produce), recurring in a subtler form. Separately,
  the "1.06 → 0.79, ~26%" headline was measured at `alpha=30`, the single
  best point in the sweep — but the value that actually ships is `alpha=100`
  (`engine/recommend.py` line 214). **Fix, and something better than what
  was proposed:** re-derived the deployed model's actual `alpha=100` score
  directly from `alpha_cv_results.csv` (RMSE 0.8101, already in that file —
  no new run needed) rather than reaching for a sweep-variant model's score
  (`cascade-2026-09-09/sanity_check.csv`'s `k=64` rows come from a
  *separately-trained* `hybrid_mf_artifacts_k64.npz`, not the deployed
  untagged model, even though both are `k=64` — verified this distinction
  before using either number). Corrected headline: **1.06 → 0.81, ~23.8%
  (~24%)**, deployed-default-consistent throughout. Added the "best-case per
  profile size, not a single configuration" caveat explicitly. Updated
  README.md's Results section to match — it now shows both the deployed-default
  comparison (24%) and the best-tested-alpha number (0.79 at alpha=30) side
  by side, labeled as what each actually is.
- **Bullet 5 mislabeled a host-memory OOM as CUDA, and cited evidence that
  belongs to a different fix.** `train_hybrid_mf.py`'s tuples→NumPy-arrays
  fix (`load_data`, lines 195–201) is host RAM — those arrays are never
  moved to `device` as a whole, only batch-sized slices are — never CUDA.
  Only `eval_holdout.py`'s 74GB tensor allocation is genuinely
  `torch.cuda.OutOfMemoryError`. Worse: the "confirmed by an actual OOM kill
  at epoch 5" quote the bullet's grounding cited for the tuples→arrays fix
  actually documents a *third*, separate fix a few dozen lines later (the
  `perm`-reshuffling change, avoiding a fresh full-array copy every epoch
  that fragmented Windows' allocator) — borrowed evidence, same category of
  problem as bullet 3's original misattribution. **Fix:** reworded to "two
  out-of-memory failures — one host-memory, one CUDA," correctly labeled
  each, and either attached the right evidence or dropped it rather than
  reattaching it to the wrong fix. `RESUME_BULLETS.md`'s grounding now
  explicitly flags the third (unclaimed) fix so it isn't confused with the
  two the bullet describes.

Both fixes are logged in `RESUME_BULLETS.md` directly (v3), with the same
inline "Grounding" convention as before.

## 2026-09-09 (part 7) — Round 3: converged

A third independent subagent re-verified round 2's two fixes from scratch
(not by re-reading round 2's own account) and did a full fresh sweep across
all 5 bullets, spot-checking the ones no round had touched (1, 2, 4) rather
than assuming two clean passes meant they'd stay clean. Both round-2 fixes
confirmed FULLY FIXED — including independently proving the
`alpha_cv_results.csv` vs `sanity_check.csv` "different trained model
instance" distinction empirically (the two files' nominally-matching
`k=64, alpha=1.0` rows give different RMSEs — 1.0632 vs 1.0607 — proving
they're not the same model read twice), not just by reading the code.

One real, if minor, thing found: "rarely `k=64` wins" understated the
actual data — `k=64` (the deployed value) wins **0 of 9** profile sizes in
the cascade grid, not "rarely." Fixed to state the true count plainly. This
makes the caveat more self-critical, not less accurate.

Also surfaced, correctly judged not to be a truthfulness defect: most of
this session's changes (`RESUME_BULLETS.md`, this file, `README.md`,
`engine/recommend.py`'s `alpha=100.0`, and every new script) are
uncommitted on `hybrid-mf-explicit-regression` — the bullets accurately
describe the current working tree, but the last actual commit (`f849ce6`)
still shows `alpha=10.0`. Not fixed here (a housekeeping/commit decision
for the repo owner, not a content correction) — flagged to the user
directly instead.

**Verdict: three independent interview passes have now converged. The
bullets in `RESUME_BULLETS.md` are genuinely interview-ready.**
