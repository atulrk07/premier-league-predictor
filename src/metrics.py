"""Probability baselines and scores. Every probability matrix uses H, D, A order.

Run `python -m src.metrics` for a training-only demonstration and report.
No validation/test outcome evaluation or learned model fitting happens here.
"""

import json

import numpy as np
import pandas as pd

from src.data import ODDS, ROOT, SEASONS, load_data, valid_odds

CLASS_ORDER = ("H", "D", "A")
TRAIN_SEASONS = tuple(SEASONS[:9])
LOG_LOSS_EPSILON = 1e-15


def _probabilities(values):
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or len(probabilities) == 0:
        raise ValueError("Expected a nonempty (n_matches, 3) probability matrix in H D A order")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("Probability rows must sum to 1")
    return probabilities


def _labels(values):
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) == 0 or pd.isna(labels).any() or not np.isin(labels, CLASS_ORDER).all():
        raise ValueError("Expected nonempty labels containing only H, D, A")
    return np.array([CLASS_ORDER.index(label) for label in labels])


def _score_inputs(y_true, probabilities):
    labels, probabilities = _labels(y_true), _probabilities(probabilities)
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probability rows must have the same length and fixture order")
    return labels, probabilities


def reorder_probabilities(probabilities, classes):
    """Map explicitly named source columns (e.g. A D H) into H D A order."""
    classes = tuple(classes)
    if len(classes) != 3 or set(classes) != set(CLASS_ORDER):
        raise ValueError("Source classes must contain H, D, A exactly once")
    probabilities = _probabilities(probabilities)
    return probabilities[:, [classes.index(label) for label in CLASS_ORDER]]


def training_frequency(train_matches):
    """Fit fixed H D A frequencies; refuse any validation/test/unknown season."""
    if not {"Season", "FTR"}.issubset(train_matches.columns):
        raise ValueError("Training frequency requires Season and FTR columns")
    if not train_matches.Season.isin(TRAIN_SEASONS).all():
        raise ValueError("Training frequency accepts only 2014/15–2022/23 rows")
    labels = _labels(train_matches.FTR)
    return np.bincount(labels, minlength=3) / len(labels)


def bookmaker_probabilities(matches):
    """Normalize inverse B365H/D/A odds; invalid triplets stay entirely missing.

    Preserve fixture indices for later matched-row comparisons. This removes the
    quoted margin proportionally; it does not establish true fair probabilities.
    """
    eligible = valid_odds(matches)
    result = pd.DataFrame(np.nan, index=matches.index, columns=list(CLASS_ORDER))
    odds = matches.reindex(columns=ODDS).apply(pd.to_numeric, errors="coerce")
    inverse = 1 / odds.loc[eligible]
    normalized = inverse.div(inverse.sum(axis=1), axis=0)
    result.loc[eligible] = normalized.to_numpy()
    return result


def multiclass_log_loss(y_true, probabilities):
    """Mean -ln(probability of actual outcome); lower is better.

    Natural logarithms. Floor only the actual outcome's probability at 1e-15 to
    keep a zero-probability mistake finite. Valid probability rows are not refit,
    rescaled, or smoothed; certainty on the correct outcome scores exactly zero.
    """
    labels, probabilities = _score_inputs(y_true, probabilities)
    actual = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.maximum(actual, LOG_LOSS_EPSILON)).mean())


def ranked_probability_score(y_true, probabilities):
    """Mean of the two squared cumulative errors (after H and after H+D).

    H D A is the PRD's fixed ordering; RPS depends on this debated ordering.
    Correct certainty scores 0; certainty on H when A occurs scores 1.
    """
    labels, probabilities = _score_inputs(y_true, probabilities)
    observed = np.eye(3)[labels]
    cumulative_error = probabilities.cumsum(axis=1)[:, :2] - observed.cumsum(axis=1)[:, :2]
    return float(np.square(cumulative_error).mean())


def baseline_report(train_matches):
    """Training illustration only: these scores are not held-out evidence."""
    frequency = training_frequency(train_matches)
    fixed = np.tile(frequency, (len(train_matches), 1))
    bookmaker = bookmaker_probabilities(train_matches)
    eligible = bookmaker.notna().all(axis=1).to_numpy()
    labels = train_matches.FTR.to_numpy()
    rows = []
    for name, subset, probabilities in [
        ("training_frequency", "all_training", fixed),
        ("training_frequency", "training_with_valid_odds", fixed[eligible]),
        ("bookmaker", "training_with_valid_odds", bookmaker.to_numpy()[eligible]),
    ]:
        selected_labels = labels if subset == "all_training" else labels[eligible]
        rows.append({
            "baseline": name, "subset": subset, "matches": len(selected_labels),
            "log_loss": multiclass_log_loss(selected_labels, probabilities) if len(selected_labels) else None,
            "rps": ranked_probability_score(selected_labels, probabilities) if len(selected_labels) else None,
        })
    hand_probs = bookmaker_probabilities(pd.DataFrame([dict(B365H=2, B365D=3, B365A=4)])).to_numpy()
    return {
        "scope": "Training illustration only, not held-out evaluation; validation/test not loaded.",
        "training_seasons": sorted(train_matches.Season.unique().tolist()),
        "class_order": list(CLASS_ORDER), "log_loss_epsilon": LOG_LOSS_EPSILON,
        "training_matches": len(train_matches),
        "training_counts": train_matches.FTR.value_counts().reindex(CLASS_ORDER, fill_value=0).to_dict(),
        "training_probabilities": dict(zip(CLASS_ORDER, frequency.tolist())),
        "bookmaker_matches": int(eligible.sum()), "bookmaker_coverage": float(eligible.mean()),
        "scores": rows,
        "hand_example": {
            "kind": "Synthetic illustration", "odds_h_d_a": [2, 3, 4], "actual": "D",
            "probabilities_h_d_a": hand_probs[0].tolist(),
            "log_loss": multiclass_log_loss(["D"], hand_probs),
            "rps": ranked_probability_score(["D"], hand_probs),
        },
    }


def main():
    train, _ = load_data(seasons=TRAIN_SEASONS)
    report = baseline_report(train)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/baseline_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Training-only demonstration; no held-out scores.")
    print("Fixed H D A probabilities:", report["training_probabilities"])
    print(f"Bookmaker coverage: {report['bookmaker_matches']}/{report['training_matches']}")
    print(pd.DataFrame(report["scores"]).to_string(index=False))
    print("Hand-calculated example:", report["hand_example"])


if __name__ == "__main__":
    main()
