# Premier League Predictor

Milestones 1–5: validated CSVs, baselines, date-safe features, eight model
comparisons, and final-test evaluation. Logistic regression C=0.1 remains the
validation-selected winner; the final test has now been evaluated and cached.
The supplied root PRD is version 2.0; it was copied to the requested [docs/PRD.md](docs/PRD.md).

## Reproduce

Tested with Python 3.12.1 on Apple Silicon macOS. Exact installed versions,
including transitive dependencies, are pinned in `requirements.txt`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# On Apple Silicon only, if XGBoost cannot find libomp:
python scripts/setup_macos_openmp.py
python -m ipykernel install --prefix .venv --name premier-league-predictor --display-name 'Premier League Predictor (.venv)'
python -m src.data --download
python -m src.metrics
python -m src.features
# Before the first final-test evaluation, fit and freeze with:
# python -m src.experiment
# The existing snapshot is already evaluated; this displays cached results:
python -m src.evaluation
python -m pytest -q
```

The OpenMP helper downloads a pinned, SHA256-checked LLVM runtime from Anaconda's
macOS ARM package archive into `.venv/openmp`, retaining its license. It adds that
local search path to the installed XGBoost library. It requires Apple's
`install_name_tool` and `otool`; rerun it if the virtual environment is recreated,
moved, or XGBoost is reinstalled. A system `brew install libomp` is another option
on machines with Homebrew. This machine had neither Homebrew nor libomp.
`certifi` supplies trusted CA certificates for Python HTTPS; TLS verification stays enabled.

After the first download, run `python -m src.data` offline. Raw CSVs and `.venv`
are ignored by Git; preserve the local raw files for exact snapshot replay.
If a fresh download differs from the recorded hash, loading fails: review the
provider correction and explicitly version a new manifest rather than silently
changing the experiment's data.

Open `notebooks/analysis.ipynb` in VS Code and select `.venv/bin/python` using
Select Kernel → Python Environments, or select the registered project kernel.
Run all cells using the saved experiment and evaluation artifacts. The notebook
displays training, validation, and cached final-test results and verifies saved
pipeline predictions on validation. Earlier all-season structural summaries
come from the milestone 1 report; notebook reruns do not open the final-test CSVs
or rerun model selection. After final-test access, the training command refuses
to overwrite this experiment. Any new fitting belongs to a separately preserved
experiment and must disclose that these test outcomes have already been seen.

## What the loader does

`src/data.py` downloads division E0 for 2014/15–2025/26, validates before caching,
and checks SHA256 against `data/sources.json` every time it loads. Each manifest
entry records the URL, raw filename, retrieval time in UTC, and hash of the exact
downloaded bytes. `reports/data_summary.json` records actual counts, per-season
column lists, exclusions, date ranges, odds availability, and a training example.

Dates use explicit day-first formats. July through August of the following year
is allowed to accommodate the COVID-delayed 2019/20 season. Required columns are
`Date`, `HomeTeam`, `AwayTeam`, `FTR`, `FTHG`, `FTAG`; `Season` is added by the loader.
Scores must be finite nonnegative integers, and H/D/A must agree with the score.
Teams must be present and different; division must be E0 when provided.
The fixture key is season, date, home team, away team. Fully blank rows and
fully identical duplicates are recorded exclusions; conflicting duplicate
fixtures or malformed match rows fail validation rather than being silently dropped.
No arbitrary upper score cap is used.

Valid Bet365 coverage requires all three decimal odds to be finite and greater
than 1. Missing odds do not remove model-eligible matches. Columns from different
seasons are combined by name; absent optional columns remain missing.
Training frequencies use 2014/15–2022/23 only. This milestone does not explore
validation or test outcome frequencies.

Data attribution: [Football-Data England archives](https://www.football-data.co.uk/englandm.php)
and [provider column notes and acknowledgements](https://www.football-data.co.uk/notes.txt).
Historical snapshots may incorporate provider corrections made after a fixture;
this limits claims about replaying exactly what was available at the time.

## Learning checkpoint

Verified snapshot: 4,560 matches across 12 seasons, 380 per season. Training has
3,420 rows, validation 380, and final test 760. The 2014/15 archive has 381 raw
rows including one fully blank row; every other archive has 380 raw rows.
No identical or conflicting fixture duplicates were found. All 4,560 matches
have valid B365H/B365D/B365A triplets. Raw schemas range from 62 to 132 columns;
the merged frame has 184 columns including the added Season column. Complete
column names are in the notebook output and `reports/data_summary.json`.

Training-only counts: H 1,533 (44.82%), D 804 (23.51%), A 1,083 (31.67%).
Example: Arsenal hosted Crystal Palace on August 16, 2014, winning 2–1 (`FTR=H`).
All seven requested packages imported successfully, and all four notebook code
cells executed sequentially with saved outputs. Run the focused pytest suite
using the reproduction command above.

One row is one match. `FTHG` and `FTAG` give full-time goals, and `FTR` identifies
the home win (H), draw (D), or away win (A). We check that the label agrees with
the scores before trusting it. These columns cannot be inputs for predicting
that same match because they reveal its result.

Answer the two questions in the notebook, then record your explanation in
`learning-notes.md` before advancing to the next milestone.

## Milestone 2: baselines and scores

Run `python -m src.metrics` to regenerate `reports/baseline_summary.json`.
That command reads only the nine training CSVs, with manifest/hash verification.
`training_frequency` refuses validation/test seasons and divides training H/D/A
counts by the training match count. Its fixed probability vector is
`[0.4482456140, 0.2350877193, 0.3166666667]` for every future fixture. Zero-count
classes retain zero probability; log loss handles zeros with its documented floor.

`bookmaker_probabilities` converts B365H/B365D/B365A by dividing each inverse
decimal odd by the sum of the three inverse odds. Returned columns are H/D/A
with the original fixture index retained; invalid triplets are entirely NaN.
It shares the loader's valid-odds rule. Missing-odds matches remain eligible
for the model dataset, and matched comparisons use the same valid-odds subset
for every baseline. No closing odds extension is implemented.

`multiclass_log_loss` averages `-ln(p_actual)`, flooring the actual outcome
probability at `1e-15`. `ranked_probability_score` averages the two squared
cumulative probability errors after H and after H+D. Perfect correct certainty
has RPS 0, while certainty on H when A occurs has RPS 1. Both metrics require
nonempty, finite, nonnegative probability rows that sum to one and H/D/A labels
aligned in fixture order; invalid inputs fail instead of being silently repaired.
`reorder_probabilities` maps explicitly named source probability columns into
H/D/A order, avoiding confusion with an estimator's alphabetical A/D/H order.

Training illustration (3,420 matches; 100% valid Bet365 coverage):

| Baseline | Log loss | RPS |
| --- | ---: | ---: |
| Fixed training frequencies | 1.064174 | 0.231855 |
| Normalized Bet365 probabilities | 0.959399 | 0.195566 |

These are training scores, not held-out results or evidence about a future
trained model. Validation and test outcomes are not scored. Normalization
removes the quoted margin proportionally, without establishing true fair
probabilities; RPS depends on the debated H/D/A ordering and remains secondary.

Hand example (synthetic draw): odds `[2,3,4]` invert to `[1/2,1/3,1/4]`, sum
to `13/12`, and normalize to `[6/13,4/13,3/13]`. Log loss is
`-ln(4/13) = 1.178654996`. RPS is
`[(6/13-0)^2 + (10/13-1)^2]/2 = 45/338 = 0.133136095`.
The notebook displays these steps and checks the scalar hand answers.
Focused tests also compare log loss with scikit-learn, verify class ordering,
reject held-out data in the frequency fit, and confirm that missing odds affect
only the matched comparison. Stop after this milestone's checkpoint.

## Milestone 3: rolling pre-match features

Run `python -m src.features` to build the 3,420-row training feature matrix,
trace Arsenal–Hull on October 18, 2014, and save these artifacts:

- `reports/features_train.csv`: fixture keys followed by 14 raw numeric features.
- `reports/feature_trace.json`: the real fixture's contributing histories and features.
- `reports/feature_summary.json`: the allowlist, missing counts, time rules, and source hashes.

`src/features.py` returns exactly `FEATURE_COLUMNS`, seven columns for each
of `home` and `away`: points means over 5 and 10 preceding matches, goals for
and against means over 5, history counts over 5 and 10, and rest days capped at
60. Win/draw/loss contributes 3/1/0 points from each team's perspective.
Both home and away appearances count. Each average uses the actual available
matches; a window is not padded to five or ten with zeros.

The builder sorts fixtures by date, then uses two passes per date: compute every
fixture's features, then add that date's results to history. History is shared
across seasons and limited to `D-365 <= previous date < D`; exactly 365 days is
eligible. No eligible history means zero counts and missing means/rest.
The function restores the input row order and index, keeping fixtures aligned.
Later validation/test feature construction must include the preceding history
rows in its input rather than call the builder on an isolated season. No real
validation/test features are built or explored during this milestone.

`trace_fixture` uses the same calculation and lists the contributing last ten
matches, marking those in the five-match window. It ignores fixtures after its
target date. The returned feature matrix has no fixture keys, outcomes, current
goals, odds, shots, cards, or half-time statistics. Fixture keys in the exported
CSV are for joining and inspection and must not become model inputs. The later
classifier pipelines will select the explicit allowlist and fit imputation,
missing indicators, and scaling on training data only; none are fitted here.

For Arsenal before October 18, the last five points are `[1,1,3,1,0]`, so the
mean is `6/5 = 1.2`. Goals scored are `[1,2,3,1,0]` (`7/5 = 1.4`); conceded are
`[1,2,0,1,2]` (`6/5 = 1.2`). Seven preceding fixtures are available in the longer
window, so its ten-match points mean is `10/7`, with counts 5 and 7. Its previous
fixture was October 5, giving 13 rest days. Hull's corresponding seven features
are `1.0`, `9/7`, `1.8`, `2.0`, `5`, `7`, and `14`.

`tests/test_features.py` uses hand-computed synthetic histories to verify the
windows, cold starts, the 365-day boundary, season carryover, and rest cap.
Score mutations check current, future, and same-day leakage, including a
deliberate shared-team same-day case. Further checks verify row-order invariance,
the feature allowlist, trace consistency, and use of earlier synthetic test
results for later fixtures. The notebook separately checks all 14 real-match
values and performs three score mutations on the real training snapshot.
Stop before model training and answer the milestone 3 checkpoint in the notebook.

## Milestone 4: eight fits and validation selection

Milestone 3's nine feature tests, all 14 real-match hand values, and the three
real-data leakage mutations passed before fitting. Run `python -m src.experiment`
to reproduce exactly the presets in `config/experiment.json`. This command opens
only 2014/15–2023/24 CSVs. Training contains 3,420 matches (August 16, 2014–May 28,
2023), and validation contains 380 (August 11, 2023–May 19, 2024). Both test CSVs
remain unopened by the milestone 4 command. Milestone 5 supplies the separate
final-test command described below.

Features are built with continuous earlier-date history before the season split;
earlier validation outcomes may inform later validation histories. Pipelines
select the 14-column allowlist, learn median imputation and missing indicators
only from training rows, and add training-fitted standard scaling for logistic
regression. Each model receives 24 inputs: 14 raw numeric features plus ten
missing indicators. There is no random split, oversampling, early stopping,
calibration fitting, or refit on validation. Seed 42 and one thread are fixed.

`fit(X_train, y_train)` learns preprocessing statistics and classifier parameters.
`predict_proba(X_validation)` applies those fixed statistics and parameters,
without learning from validation labels. H/D/A labels are encoded 0/1/2, and
estimator probability columns are explicitly mapped back into H/D/A order.
XGBoost's float32 row-sum rounding is normalized only within a 1e-6 tolerance.

Actual comparison on the same 380 validation fixtures:

| Configuration | Train log loss | Validation log loss | Validation RPS | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| **LR C=0.1 — winner** | 1.002021 | **0.966052** | 0.201246 | 55.00% |
| LR C=1.0 | 1.001948 | 0.966204 | 0.201214 | 55.00% |
| RF depth=5, leaf=10 | 0.988530 | 0.985354 | 0.208660 | 52.37% |
| RF depth=8, leaf=10 — family selection | 0.946966 | 0.981781 | 0.207153 | 53.16% |
| RF depth=8, leaf=20 | 0.968427 | 0.982697 | 0.207585 | 52.11% |
| XGB depth=2, rounds=200, rate=0.05 — family selection | 0.966713 | 0.976858 | 0.205072 | 51.84% |
| XGB depth=3, rounds=200, rate=0.05 | 0.919902 | 0.979498 | 0.205206 | 53.16% |
| XGB depth=3, rounds=400, rate=0.03 | 0.905799 | 0.978959 | 0.204775 | 54.21% |
| Fixed training-frequency reference | 1.064174 | 1.054158 | 0.233777 | 46.05% |

Every RF uses 300 trees. `C` is inverse regularization strength (smaller means
stronger shrinkage). `max_depth` limits tree complexity; `min_samples_leaf`
requires a minimum number of training rows per leaf. Random Forest averages
independently fitted bootstrap trees. XGBoost builds successive rounds to reduce
remaining loss; `learning_rate` shrinks each round's contribution. Its
`subsample=0.8` and `colsample_bytree=0.8` sample rows and features during tree
construction. Multiclass boosting rounds can create multiple trees per round.
Logistic regression uses lbfgs with a 2,000-iteration convergence budget.

Select the minimum validation log loss within each family, then apply the
0.005 tolerance: prefer logistic regression, then RF, then XGBoost among eligible
family selections. Here only LR C=0.1 is within tolerance of the minimum, so it
wins directly. The reference baseline is not a searched candidate. These are
validation results and do not establish final-test or bookmaker performance.

Saved artifacts:

- `reports/validation_results.csv`: eight configurations, metrics, family selections, winner.
- `reports/frequency_validation.json`: training-only frequencies and reference scores.
- `reports/validation_predictions.csv`: fixture keys, labels, split, candidate IDs, H/D/A probabilities.
- `reports/experiment_config.json`: configuration snapshot, versions, source hashes, split dates, and preprocessing audits.
- `reports/winner_decision.json`: selection rule, selected candidates, winner, timestamp, artifact hash, and no-test/no-refit status.
- `reports/selected_pipelines.joblib`: the three selected training-fitted pipelines and frequency baseline.
- `reports/validation_example.json`: one actual prediction and its raw inputs.

Burnley–Man City on August 11, 2023 receives H 11.95%, D 14.70%, A 73.35% from
the winner; both the highest-probability prediction and actual outcome are A.
Burnley's zero eligible EPL history is handled with training medians and missing
indicators. The notebook reproduces the saved pipelines' validation losses and
verifies prediction does not alter preprocessing statistics. Synthetic tests
protect split/fitting boundaries, class mapping, exact presets, and the selection
tolerance, including its 0.005 boundary. Answer the two notebook questions and
stop before final-test evaluation.

## Milestone 5: final test and calibration inspection

The first `python -m src.evaluation` run verifies the frozen pipeline and config
hashes, source provenance, and package versions before opening final-test CSVs.
It loads the complete history to build strictly date-safe features, predicts
with the three saved family selections, and uses the saved training-frequency
vector. Neither model parameters nor preprocessing statistics are fitted again.
The original winner decision and pipeline artifact remain unchanged.

The evaluation command writes a timestamped started/completed record and hashes
of the output artifacts. Completed reruns verify and display the cached results;
they do not open test CSVs or recompute test predictions. Training refuses to
overwrite the frozen experiment once the final-test record exists. The historical
`test_evaluated: false` in `winner_decision.json` describes the pre-test freeze;
current status is in `test_evaluation.json`.

Bookmaker coverage is **380/380 in 2024/25, 380/380 in 2025/26, 760/760 combined**.
All-row model scores therefore equal the matched-bookmaker scores here. The
implementation and synthetic tests also handle incomplete odds by comparing
every model and baseline on the identical valid-odds fixture subset.

| Forecast | 2024/25 log loss | 2025/26 log loss | Combined log loss | Combined RPS | Combined accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Frozen winner: LR C=0.1** | 1.014637 | 1.057667 | **1.036152** | 0.214559 | 48.82% |
| RF depth=8, leaf=10 | 1.039469 | 1.058497 | 1.048983 | 0.219184 | 48.03% |
| XGB depth=2, rounds=200, rate=0.05 | 1.041766 | 1.064878 | 1.053322 | 0.219794 | 47.24% |
| Training frequency | 1.081070 | 1.083292 | 1.082181 | 0.231530 | 41.71% |
| Normalized bookmakers | 0.970758 | 1.018547 | 0.994653 | 0.201491 | 51.45% |

The frozen winner improves on the frequency reference by 0.0460 log loss but
trails bookmakers by 0.0415 on these matches. No winner is reselected using test
results, and no claim of betting profit or universal performance is made.

Calibration is inspected with three one-vs-rest curves and five equal-width bins
per outcome. Count bars show empty and small bins. In the home 0.6–0.8 bin,
105 fixtures have mean forecast 0.6658 and observed frequency 0.6000, suggesting
overprediction in that bin. The highest home bin has only five observations,
so its observed frequency of 1.0 is noisy. The main draw bin contains 683
fixtures: mean probability 0.2432 versus observed frequency 0.2592. This inspection
does not fit a calibrator; a future calibration experiment needs a separate
partition and must disclose reuse of the already-opened test outcomes.

Descriptive error analysis uses a predeclared low-history definition: either
team has fewer than five eligible prior matches. Log loss is 1.157225 over 28
low-history fixtures and 1.031521 over 732 full-history fixtures. Six zero-history
fixtures are a subset of the 28, not an additional group to add to them.
The winner predicted a draw once, correctly: precision 1/1 (100%), recall
1/197 (0.51%). High precision based on one prediction is weak evidence; the model
missed 196 draws as its highest-probability outcome. It nevertheless assigns a
draw probability to every fixture, which contributes to probability metrics.
Groups and worst errors are post-test description, not new tuning or selection.

Example: Man United–Fulham, August 16, 2024 receives H 41.65%, D 21.87%, A 36.47%.
The actual result was H; its individual log loss is 0.875827. Correct winner
picking does not imply certainty or zero log loss.

Saved outputs:

- `reports/test_results.csv`: all-row and matched scores by season and combined, with denominators and correct counts.
- `reports/test_coverage.csv`: bookmaker eligibility counts and fractions.
- `reports/test_classification.csv`: H/D/A precision, recall, F1, support, and predicted counts.
- `reports/test_predictions.csv`: 3,800 forecast rows (five forecasts × 760 fixtures), fixture IDs, labels, splits, probabilities, and odds-valid flags.
- `reports/calibration_bins.csv`, `calibration.png`, `calibration.pdf`: five bins per outcome, counts, and exportable figures.
- `reports/test_error_groups.csv`, `test_error_examples.csv`: descriptive groups and the five largest winner losses.
- `reports/test_example.json`: one real prediction, features, and loss.
- `reports/test_evaluation.json`: timestamps, frozen selection identity, source hashes, output hashes, and no-refit/no-calibrator status.

See `src/evaluation.py`, `tests/test_evaluation.py`, and the milestone 5 notebook
section. Stop here and answer the two understanding questions before milestone 6.
