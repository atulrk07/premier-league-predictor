# Learning notes

## Milestone 1 — data loading

1. Why would using a fixture's FTHG or FTAG as inputs leak the answer?

   Atul's answer: The match score tells the model who won. Correct: full-time
   goals reveal the target and are unavailable before the match. Earlier matches'
   scores can contribute to future pre-match features.

2. Should a valid match with missing bookmaker odds be removed from model data? Why?

   Atul's initial answer: Yes, without bookmaker odds the model cannot train or
   test uniformly.

   Clarification: Keep the match for model training/testing. Bookmaker odds are
   excluded from model inputs; they are used only for the separate bookmaker
   benchmark. For a fair comparison, score every model and baseline on the same
   subset of fixtures with valid odds, and report that subset's size and coverage.

Checkpoint: answer 1 is correct; the clarification for answer 2 was explained
again with separate model evaluation and bookmaker comparison examples.
Atul requested milestone 2 before restating the clarification.

## Milestone 2 — baselines and metrics

Implemented at Atul's request. Answers and clarification:

1. Why must the frequency baseline use training outcomes only, rather than
   including validation or test outcomes?

   Atul identified the need to separate training from validation and testing.
   Clarification: calculate the frequencies on training outcomes only, then use
   those fixed probabilities to predict validation/test fixtures. Including their
   outcomes in the frequency calculation would leak their answers.

2. In the synthetic draw example, why does log loss use 4/13, and what happens
   to the loss if the draw probability becomes much smaller?

   Atul correctly identified 4/13 as the normalized draw probability, but initially
   thought the loss would decrease if that probability fell. Clarification: loss
   increases because the draw actually happened; -ln(p_actual) penalizes assigning
   less probability to the observed outcome. A draw probability of 0.10 gives
   loss 2.303, larger than 1.179 at 4/13.

Reference hand calculation: odds 2, 3, 4 normalize to H/D/A probabilities
6/13, 4/13, 3/13. For an actual draw, log loss is -ln(4/13) = 1.178654996;
RPS is [(6/13)^2 + (10/13 - 1)^2]/2 = 45/338 = 0.133136095.
Training-only scores are illustrations, not held-out evaluation. Atul subsequently
requested the feature milestone. No classifier training has started.

## Milestone 3 — date-safe rolling features

Seven features per team give 14 raw inputs. Compute all fixtures on a date before
adding that day's results. Eligible history has date in [D-365, D); it carries
across seasons. No history means zero counts and missing means/rest.

Real trace: Arsenal–Hull, October 18, 2014. Arsenal's last-five points are
1, 1, 3, 1, 0, giving 1.2 points per match. Its ten-match window has only seven
available fixtures and 10 points, so the average is 10/7; counts are 5 and 7.
Its scoring means are 1.4 for and 1.2 against; rest is 13 days. Full histories
and both teams' values are in the notebook and reports/feature_trace.json.

Answers:

1. Why are Arsenal's history counts 5 and 7, and why divide the longer-window
   points total by 7 rather than 10?

   Atul's answer: They only had seven matches, not ten. Correct: use the actual
   eligible history count as the denominator rather than padding missing matches.

2. Why compute every fixture's features on a date before adding any results
   from that date?

   Atul's answer: We only use features from before the date. Correct: use results
   from strictly earlier dates. Another fixture on the same date is excluded,
   even if it kicked off earlier, so today's outcomes cannot leak into inputs.

Milestone 3 understanding checkpoint completed. Stopped before model training
until Atul explicitly requested milestone 4.

At the end of milestone 3, no model training, imputation, or scaling had been performed.

## Milestone 4 — training and validation selection

Milestone 3's nine feature tests and real-match score-mutation checks passed
before fitting. Eight PRD configurations were fitted on 3,420 training matches
and scored on 380 validation matches. No final-test CSV was opened by the
experiment. The feature inputs grow from 14 to 24 after ten missing indicators.
Medians, indicators, logistic scaling, and classifier parameters are fitted on
training data only.

Winner: logistic regression C=0.1, validation log loss 0.966052, versus 1.054158
for the training-frequency reference. Selected family candidates are RF depth 8,
minimum leaf 10, and XGB depth 2, 200 rounds, learning rate 0.05. Neither is within
0.005 of the minimum. All three selected pipelines and the winner decision are
saved; none are refitted on validation.

`fit` learns parameters and preprocessing. `predict_proba` reuses them without
learning and returns H/D/A probabilities. Burnley–Man City, August 11, 2023:
H 11.95%, D 14.70%, A 73.35%; the actual outcome was A.

Pending your answers:

1. What does fit learn, and what remains fixed when predict_proba runs on
   validation matches?
2. Why did LR C=0.1 win, and how would the 0.005 rule apply if Random Forest
   had a slightly lower validation log loss?

Stopped before final-test evaluation until Atul requested milestone 5. The code
walkthrough covered feature selection, encoding, splits, preprocessing, fitting,
probability mapping, selection, and an actual validation prediction. Atul did
not submit answers to the two milestone 4 checkpoint questions before proceeding.

## Milestone 5 — frozen final-test evaluation

Explicitly requested by Atul. Evaluated the frozen selections on 2024/25 and
2025/26, 380 fixtures each. Bookmaker coverage is 760/760. No models, imputation,
scaling, or calibrator were fitted; the original winner decision is preserved.

Combined log loss: winner 1.036152, frequency 1.082181, bookmakers 0.994653.
The winner beats the frequency baseline on these matches but trails bookmakers.
Calibration plots show counts for five bins per outcome; tiny bins are noisy.

The winner predicted one draw and got it right: precision 100% (1/1), but recall
0.51% (1/197). Low-history fixtures (either team with fewer than five prior
matches) have log loss 1.157225 across 28 fixtures, versus 1.031521 across 732
full-history fixtures. These are descriptive post-test observations, not tuning.

Example: Man United–Fulham, August 16, 2024: H 41.65%, D 21.87%, A 36.47%; actual
H; single-match log loss 0.875827.

Pending your answers:

1. Why must model and bookmaker comparisons use identical valid-odds fixtures,
   even though this snapshot has 100% coverage?
2. What do draw precision 100% and recall 0.51% mean, and why does the precision
   figure alone not establish good draw forecasting?

Stopped after milestone 5. Final-test outcomes are now seen; later experiments
must disclose their reuse. Cached reports can be displayed without reevaluating.
