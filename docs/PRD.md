# Premier League ML Predictor

Predict Premier League home-win, draw, and away-win probabilities from pre-match results and scoring form. Compare logistic regression, Random Forest, and XGBoost against training-outcome frequencies and normalized bookmaker odds using chronological validation and final-test seasons.

## Scope

The implementation includes validated historical data, date-safe rolling features, eight model configurations, frozen model selection, final-test evaluation, calibration plots, and prediction exports. Presentation uses reusable Python modules, one notebook, saved reports, and reproduction instructions in the README.

API and frontend development, Docker, CI, load testing, live feeds, and deployment are outside scope. No probability calibrator is fitted.

## Data contract

Use [football-data.co.uk](https://www.football-data.co.uk/englandm.php) Premier League CSVs, division `E0`, for 2014/15–2025/26. Source URLs follow `https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv`, with codes `1415` through `2526`.

- Cache raw bytes in `data/raw/`; record URLs, retrieval timestamps, filenames, and SHA256 hashes in `data/sources.json`. Reject hash mismatches rather than silently replacing snapshots.
- Require `Date`, `HomeTeam`, `AwayTeam`, `FTR`, `FTHG`, and `FTAG`. Parse day-first dates and validate season boundaries, teams, division, nonnegative integer scores, and score/result consistency.
- Identify fixtures by season, date, home team, and away team. Remove blank rows and identical duplicates; reject conflicting duplicates. Record columns, counts, exclusions, and date ranges.
- Missing or invalid odds reduce bookmaker benchmark coverage only; retain those matches for model evaluation. Provider corrections and odds collection times limit historical replay fidelity.

The recorded snapshot contains **4,560 validated matches**, with 380 per season. One blank row was removed from 2014/15; no identical duplicates were removed. Details are saved in [data_summary.json](../reports/data_summary.json).

## Features and leakage prevention

Build seven numeric features for each team, with `home_` and `away_` prefixes: **14 raw features** total.

| Feature per team | Calculation |
| --- | --- |
| `points_mean_5`, `points_mean_10` | Mean points over the latest 5 or 10 eligible matches; win = 3, draw = 1, loss = 0 |
| `goals_for_mean_5` | Mean goals scored over the latest 5 eligible matches |
| `goals_against_mean_5` | Mean goals conceded over the latest 5 eligible matches |
| `history_count_5`, `history_count_10` | Actual match counts in those windows |
| `rest_days` | Days since the latest eligible match, capped at 60 |

For a fixture dated `D`, eligible history satisfies `D - 365 days <= match date < D`, including home and away appearances across seasons. Compute every fixture on a date before adding that day's results to history. Earlier validation or test results may inform later fixtures; fitted parameters remain fixed.

Use the actual available count as the mean denominator. Teams with no eligible history have zero counts and missing means/rest days. Home advantage is represented by the home/away feature columns.

Select inputs through an explicit allowlist. Exclude bookmaker odds and current-match goals, half-time statistics, shots, and cards. Fit median imputation, missing-value indicators, and logistic-regression scaling on training rows only. Preserve entirely missing feature columns. The saved pipelines have 24 processed inputs: 14 raw features and 10 missing-value indicators.

## Experimental design

| Partition | Seasons | Matches | Use |
| --- | --- | ---: | --- |
| Training | 2014/15–2022/23 | 3,420 | Fit preprocessing, models, and frequency baseline |
| Validation | 2023/24 | 380 | Select configurations and model family |
| Final test | 2024/25–2025/26 | 760 | Evaluate frozen candidates by season and combined |

Use chronological splits without random splitting or shuffled cross-validation. Fix features, metrics, split boundaries, and search budget before selection. Final-test outcomes must not influence fitting or selection.

### Model configurations

| Family | Candidates |
| --- | --- |
| Logistic regression | `C = 0.1`, `1.0`; median imputation and standard scaling |
| Random Forest | 300 trees; `(max_depth, min_samples_leaf)` = `(5, 10)`, `(8, 10)`, `(8, 20)` |
| XGBoost | `(max_depth, n_estimators, learning_rate)` = `(2, 200, 0.05)`, `(3, 200, 0.05)`, `(3, 400, 0.03)` |

Use seed 42 and one thread. Logistic regression uses `lbfgs` with `max_iter=2000`. XGBoost uses `multi:softprob`, three classes, CPU histogram trees, and row/column subsampling of 0.8. Use uniform class weights, fixed tree counts, no oversampling, and no early stopping. Full settings are stored in [experiment.json](../config/experiment.json); fitted preprocessing audits and tested package versions are recorded in `reports/experiment_config.json`.

### Selection and freezing

1. Select the configuration with the lowest validation log loss within each family.
2. Among those three candidates, consider every candidate within 0.005 of the lowest validation log loss. Prefer logistic regression, then Random Forest, then XGBoost. This tolerance is a simplicity rule, not a significance test.
3. Save all eight configurations' scores, the three fitted pipelines, and the winner decision before test evaluation. Do not refit on validation data.
4. Evaluate all three frozen candidates on identical test fixtures. Preserve the validation-selected winner regardless of test rankings. Label post-test error analysis as descriptive; subsequent experiments must disclose test reuse.

Final-test evaluation is complete. The evaluator verifies frozen artifacts and source hashes; repeat runs verify and display cached outputs. The experiment command refuses refitting once test evaluation has started.

## Metrics and baselines

Encode targets as `H=0`, `D=1`, `A=2`; explicitly map estimator probabilities into **H/D/A order**. Probability rows must be finite, within `[0, 1]`, and sum to one.

- **Log loss:** mean negative natural logarithm of the observed outcome's probability, floored at `1e-15`. Lower is better; this is the selection metric.
- **RPS:** square the forecast/observed cumulative-probability differences after H and after H+D, then average the two terms. Correct certainty scores 0; certainty on H when A occurs scores 1. RPS is secondary because its outcome ordering is debated.
- **Accuracy:** fraction of matches whose highest-probability outcome matches the result. Report correct counts and denominators.
- **Classification:** H/D/A precision, recall, F1, support, and predicted counts, including draw performance. Undefined ratios are reported as zero.

The frequency baseline uses fixed training-label proportions: H = `1533/3420`, D = `804/3420`, A = `1083/3420`.

For bookmaker odds `o_H`, `o_D`, `o_A` from `B365H`, `B365D`, `B365A`, compute `p_i = (1/o_i) / sum_j(1/o_j)`. Require all three decimal odds to be finite and greater than one. This normalization removes the overround but does not establish true fair probabilities; these columns are not necessarily opening prices.

Report model and frequency scores on all eligible test matches, then compare all three models and both baselines on the identical subset with valid odds. Publish coverage and denominators. Lower log loss alone does not establish calibration quality or betting profit.

### Recorded results

The validation-selected winner is logistic regression with `C=0.1`, validation log loss **0.966052**. It was the only family candidate within the selection tolerance. Combined test results below use all **760 matched fixtures**, with 100% bookmaker coverage:

| Forecast | Log loss | RPS | Accuracy |
| --- | ---: | ---: | ---: |
| Logistic regression, `C=0.1` | 1.036152 | 0.214559 | 48.82% |
| Random Forest, depth 8 / leaf 10 | 1.048983 | 0.219184 | 48.03% |
| XGBoost, depth 2 / 200 trees / rate 0.05 | 1.053322 | 0.219794 | 47.24% |
| Training frequency | 1.082181 | 0.231530 | 41.71% |
| Normalized bookmaker odds | 0.994653 | 0.201491 | 51.45% |

These values come from [test_results.csv](../reports/test_results.csv); per-season scores and correct counts are included there. The winner improves on training frequency but has higher test log loss than the bookmaker baseline. A single validation season and two test seasons provide limited evidence of performance stability.

## Presentation and outputs

[analysis.ipynb](../notebooks/analysis.ipynb) imports shared code, runs top to bottom without hidden state, and displays saved final-test outputs without repeating test prediction or fitting. It presents:

- Data counts, required columns, exclusions, odds availability, and one example match.
- A real fixture's feature history and calculations; all eight validation configurations with training/validation log loss, validation RPS and accuracy, the frequency baseline, and the selection decision.
- Final-test results by season and combined, with matched bookmaker comparisons and coverage.
- Three one-versus-rest reliability curves for the winner, with five equal-width bins per outcome, visible counts, and no points for empty bins. Small bins are noisy; these plots inspect calibration without fitting a calibrator.
- A real prediction and its log loss, draw performance, and errors grouped by outcome and available history. Low history means either team has fewer than five eligible prior matches; report group sizes and the zero-history subset.

Save reports and figures under `reports/`: validation tables/predictions, `winner_decision.json`, `selected_pipelines.joblib`, test scores/coverage/classification/predictions, calibration bins and PNG/PDF plots, error groups/examples, and prediction examples. Prediction exports include fixture identifiers, split, observed labels, and H/D/A probabilities. `test_evaluation.json` records evaluation status and artifact hashes; the winner decision remains the original freeze record.

## Reproducibility and acceptance criteria

Use Python with pandas, NumPy, scikit-learn, XGBoost, matplotlib, pytest, and a notebook kernel. Pin dependencies in [requirements.txt](../requirements.txt); document environment setup and commands in [README.md](../README.md). Keep code, configuration, source manifests, synthetic test fixtures, reports, and notebook outputs under version control; exclude virtual environments and raw downloads. Reproduction requires the recorded raw snapshot; provider updates must not silently change it.

Shared modules are `src/data.py` (loading/validation), `src/features.py` (feature construction/tracing), `src/metrics.py` (metrics/baselines), `src/experiment.py` (fitting/selection), and `src/evaluation.py` (frozen evaluation). Initial training/validation uses `python -m src.experiment`; final-test evaluation is a separate explicit `python -m src.evaluation` step. Existing completed evaluations are displayed from verified cached reports.

Acceptance requires:

- Validated data and provenance with actual counts, columns, exclusions, and date ranges.
- Synthetic feature calculations and a real fixture trace; current, future, and same-day score mutations must leave relevant features unchanged.
- Split and preprocessing checks proving fitting uses training rows only; explicit class mapping, known log-loss/RPS/odds calculations, and valid probabilities.
- Measured validation scores for all eight configurations, a frozen decision, and per-season/combined final-test scores for all selected families and matched baselines.
- Saved calibration and error reports with counts, prediction exports, unchanged fitted preprocessing during evaluation, and artifact-integrity checks.
- A reproducible notebook and documented commands against the recorded snapshot, with findings and limitations reported accurately. Correctness checks reside in `tests/test_correctness.py`, `tests/test_features.py`, `tests/test_experiment.py`, and `tests/test_evaluation.py`.

Optional extensions include feature ablation, permutation importance, additional chronological folds, venue-specific features, and a separate closing-odds benchmark using `B365CH/B365CD/B365CA`. Importance describes model reliance, not causal influence. A fitted calibrator requires a separate earlier calibration partition and an evaluation design fixed before test access. Changes made after opening the existing test seasons must disclose reuse and cannot be presented as another untouched evaluation.

## Technical references

- [Football Data column and odds notes](https://www.football-data.co.uk/notes.txt)
- [scikit-learn preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Probability calibration and reliability plots](https://scikit-learn.org/stable/modules/calibration.html)
- [RPS in football forecasting](https://arxiv.org/abs/1908.08980)
