"""Milestone 5: explicit final-test evaluation of frozen milestone 4 pipelines.

Run `python -m src.evaluation` once after `python -m src.experiment`.
Completed reruns display cached results instead of opening test CSVs again.
"""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import KEY, ROOT, SEASONS, load_data
from src.experiment import predict_hda
from src.features import FEATURE_COLUMNS, HISTORY_DAYS, REST_CAP_DAYS, build_features
from src.metrics import CLASS_ORDER, LOG_LOSS_EPSILON, bookmaker_probabilities, multiclass_log_loss, ranked_probability_score

TEST_SEASONS = tuple(SEASONS[10:])
CALIBRATION_EDGES = np.linspace(0, 1, 6)
LOW_HISTORY_THRESHOLD = 5
PROBABILITY_COLUMNS = ["p_H", "p_D", "p_A"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(root=ROOT):
    """Verify the pre-test freeze before allowing final-test loading."""
    root = Path(root)
    reports = root / "reports"
    decision = json.loads((reports / "winner_decision.json").read_text())
    metadata = json.loads((reports / "experiment_config.json").read_text())
    artifact = reports / "selected_pipelines.joblib"
    if file_hash(artifact) != decision["artifact_sha256"]:
        raise ValueError("Frozen pipeline artifact hash mismatch")
    if file_hash(root / "config/experiment.json") != metadata["config_sha256"]:
        raise ValueError("Configuration changed after the validation freeze")
    sources = json.loads((root / "data/sources.json").read_text())
    if any(sources[season]["sha256"] != digest for season, digest in metadata["source_sha256"].items()):
        raise ValueError("Training/validation data snapshot changed after fitting")
    if metadata["history_days"] != HISTORY_DAYS or metadata["rest_cap_days"] != REST_CAP_DAYS:
        raise ValueError("Feature definitions changed after fitting")
    if any(version(name) != expected for name, expected in metadata["package_versions"].items()):
        raise ValueError("Use the frozen experiment's tested package versions")
    bundle = joblib.load(artifact)
    if (bundle["winner_id"] != decision["winner_id"] or bundle["family_candidates"] != decision["family_candidates"]
            or tuple(bundle["feature_columns"]) != FEATURE_COLUMNS or tuple(bundle["class_order"]) != CLASS_ORDER
            or set(bundle["pipelines"]) != set(decision["family_candidates"]) or decision["refit_on_validation"]):
        raise ValueError("Frozen selection metadata is inconsistent")
    return bundle, decision, metadata, sources


def classification_summary(labels, probabilities):
    """H/D/A precision, recall and counts; undefined ratios use zero."""
    multiclass_log_loss(labels, probabilities)  # Validate labels and probability rows.
    labels = np.asarray(labels)
    predicted = np.asarray(CLASS_ORDER)[np.asarray(probabilities).argmax(axis=1)]
    rows = []
    for outcome in CLASS_ORDER:
        support = int((labels == outcome).sum())
        predicted_count = int((predicted == outcome).sum())
        true_positive = int(((labels == outcome) & (predicted == outcome)).sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        rows.append({"outcome": outcome, "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                     "support": support, "predicted_count": predicted_count, "true_positive": true_positive})
    return rows


def score_row(labels, probabilities):
    if len(labels) == 0:
        return {"matches": 0, "log_loss": np.nan, "rps": np.nan, "accuracy": np.nan, "correct_predictions": 0}
    loss = multiclass_log_loss(labels, probabilities)
    predicted = np.asarray(CLASS_ORDER)[np.asarray(probabilities).argmax(axis=1)]
    correct = int((predicted == np.asarray(labels)).sum())
    return {"matches": len(labels), "log_loss": loss, "rps": ranked_probability_score(labels, probabilities),
            "accuracy": correct / len(labels), "correct_predictions": correct}


def evaluation_tables(test_matches, forecasts, bookmaker):
    """All test rows first; then identical valid-odds rows for every forecast."""
    if not bookmaker.index.equals(test_matches.index) or list(bookmaker.columns) != list(CLASS_ORDER):
        raise ValueError("Bookmaker forecasts must align with fixture rows in H/D/A order")
    if bookmaker.isna().any(axis=1).ne(bookmaker.isna().all(axis=1)).any():
        raise ValueError("Invalid bookmaker triplets must be entirely missing")
    eligible = bookmaker.notna().all(axis=1).to_numpy()
    outcomes = test_matches.FTR.to_numpy()
    rows, reports, coverage = [], [], []
    for season in [*TEST_SEASONS, "combined"]:
        season_mask = np.ones(len(test_matches), dtype=bool) if season == "combined" else test_matches.Season.eq(season).to_numpy()
        matched = season_mask & eligible
        total, valid = int(season_mask.sum()), int(matched.sum())
        coverage.append({"season": season, "test_matches": total, "bookmaker_matches": valid,
                         "coverage": valid / total if total else np.nan})
        for subset, mask, candidates in [
            ("all_test", season_mask, forecasts),
            ("matched_bookmaker", matched, {**forecasts, "bookmaker": bookmaker.to_numpy()}),
        ]:
            for name, probabilities in candidates.items():
                probabilities = np.asarray(probabilities)
                if probabilities.shape != (len(test_matches), 3):
                    raise ValueError("Forecasts must contain one H/D/A row per test fixture")
                selected_labels, selected_probs = outcomes[mask], probabilities[mask]
                identifiers = {"season": season, "subset": subset, "forecast": name}
                rows.append({**identifiers, **score_row(selected_labels, selected_probs)})
                if len(selected_labels):
                    reports.extend({**identifiers, **row} for row in classification_summary(selected_labels, selected_probs))
    return pd.DataFrame(rows), pd.DataFrame(reports), pd.DataFrame(coverage)


def calibration_bins(labels, probabilities):
    """Three one-vs-rest curves; five bins: [0,.2], (.2,.4], ..., (.8,1]."""
    multiclass_log_loss(labels, probabilities)
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    rows = []
    for column, outcome in enumerate(CLASS_ORDER):
        values = probabilities[:, column]
        assigned = np.searchsorted(CALIBRATION_EDGES[1:-1], values, side="left")
        for index in range(5):
            mask = assigned == index
            count = int(mask.sum())
            rows.append({"outcome": outcome, "bin": index + 1, "lower": CALIBRATION_EDGES[index],
                         "upper": CALIBRATION_EDGES[index + 1], "count": count,
                         "mean_probability": float(values[mask].mean()) if count else np.nan,
                         "observed_frequency": float((labels[mask] == outcome).mean()) if count else np.nan})
    return pd.DataFrame(rows)


def plot_calibration(bins, destination, winner_id):
    # Local cache and headless plotting work in both the CLI and notebook.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/.matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
    for column, outcome in enumerate(CLASS_ORDER):
        frame = bins.loc[bins.outcome.eq(outcome)]
        occupied = frame.loc[frame["count"] > 0]
        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.plot(occupied.mean_probability, occupied.observed_frequency, "o-", label=winner_id)
        for point_index, row in enumerate(occupied.itertuples()):
            # The draw means are close together; separate their count labels.
            offset = ((-7, 10) if point_index == 0 else (7, -14)) if outcome == "D" else (4, 6)
            alignment = "right" if outcome == "D" and point_index == 0 else "left"
            ax.annotate(f"n={row.count}", (row.mean_probability, row.observed_frequency), xytext=offset,
                        textcoords="offset points", fontsize=9, ha=alignment)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{outcome} vs rest", xlabel="Mean predicted probability", ylabel="Observed frequency")
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(fontsize=8)
        bars = axes[1, column]
        bars.bar((frame.lower + frame.upper) / 2, frame["count"], width=0.17)
        for row in frame.itertuples():
            bars.text((row.lower + row.upper) / 2, row.count, str(row.count), ha="center", va="bottom", fontsize=9)
        bars.set(xlim=(0, 1), xlabel="Probability bin", ylabel="Matches")
        bars.set_ylim(0, max(1, frame["count"].max()) * 1.2)
    fig.suptitle(f"Frozen validation winner: {winner_id} — final test, five bins per outcome\nSmall bins are noisy; inspection only, no calibrator fitted")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(destination, dpi=180)
    fig.savefig(Path(destination).with_suffix(".pdf"))
    plt.close(fig)


def error_groups(test_matches, features, probabilities):
    """Descriptive post-test groups, fixed before test access; not model selection."""
    if not features.index.equals(test_matches.index):
        raise ValueError("Error-analysis features must align with fixtures")
    low = (features.home_history_count_5 < LOW_HISTORY_THRESHOLD) | (features.away_history_count_5 < LOW_HISTORY_THRESHOLD)
    zero = features.home_history_count_5.eq(0) | features.away_history_count_5.eq(0)
    groups = {
        "all": np.ones(len(test_matches), dtype=bool),
        "actual_H": test_matches.FTR.eq("H").to_numpy(), "actual_D": test_matches.FTR.eq("D").to_numpy(),
        "actual_A": test_matches.FTR.eq("A").to_numpy(),
        "low_history_either_team_lt5": low.to_numpy(), "full_history_both_5": (~low).to_numpy(),
        "zero_history_either_team": zero.to_numpy(),
    }
    rows = []
    for season in [*TEST_SEASONS, "combined"]:
        season_mask = np.ones(len(test_matches), dtype=bool) if season == "combined" else test_matches.Season.eq(season).to_numpy()
        for group, group_mask in groups.items():
            mask = season_mask & group_mask
            rows.append({"season": season, "group": group, **score_row(test_matches.FTR.to_numpy()[mask], probabilities[mask])})
    return pd.DataFrame(rows)


def fixture_ids(matches):
    keys = matches[KEY].copy()
    keys["Date"] = keys.Date.dt.strftime("%Y-%m-%d")
    return keys.astype(str).agg("|".join, axis=1)


def display_results(reports=ROOT / "reports"):
    reports = Path(reports)
    summary = json.loads((reports / "test_evaluation.json").read_text())
    print("Frozen validation winner:", summary["winner_id"])
    print(pd.read_csv(reports / "test_results.csv").to_string(index=False))
    print("Bookmaker coverage:")
    print(pd.read_csv(reports / "test_coverage.csv").to_string(index=False))
    return summary


def run_evaluation():
    bundle, decision, metadata, sources = load_frozen()  # Must precede test CSV access.
    reports = ROOT / "reports"
    marker = reports / "test_evaluation.json"
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing["frozen_artifact_sha256"] != decision["artifact_sha256"] or existing["winner_decision_sha256"] != file_hash(reports / "winner_decision.json"):
            raise ValueError("The evaluated freeze changed; do not silently replace the final test")
        if any(sources[season]["sha256"] != digest for season, digest in existing["source_sha256"].items()):
            raise ValueError("Final-test snapshot changed after evaluation")
        if existing["status"] == "complete":
            for name, digest in existing["output_sha256"].items():
                if file_hash(reports / name) != digest:
                    raise ValueError(f"Cached evaluation artifact changed: {name}")
            return display_results(reports)
    summary = {
        "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_artifact_sha256": decision["artifact_sha256"], "winner_decision_sha256": file_hash(reports / "winner_decision.json"),
        "winner_id": decision["winner_id"], "winner_family": decision["winner_family"],
        "family_candidates": decision["family_candidates"], "frozen_at_utc": decision["frozen_at_utc"],
        "source_sha256": {season: sources[season]["sha256"] for season in SEASONS},
        "test_seasons": list(TEST_SEASONS), "class_order": list(CLASS_ORDER),
        "low_history_definition": "Either team has fewer than five eligible prior EPL matches",
        "low_history_threshold": LOW_HISTORY_THRESHOLD, "calibration_edges": CALIBRATION_EDGES.tolist(),
        "calibration_intervals": "[0,.2], (.2,.4], (.4,.6], (.6,.8], (.8,1]",
        "refit": False, "calibrator_fitted": False, "zero_division_classification": 0,
        "analysis_scope": "Error groups and worst errors describe final-test outcomes; no post-test tuning or winner reselection.",
    }
    if datetime.fromisoformat(decision["frozen_at_utc"]) > datetime.fromisoformat(summary["started_at_utc"]):
        raise ValueError("Selection must be frozen before evaluation")
    marker.write_text(json.dumps(summary, indent=2) + "\n")
    matches, _ = load_data()  # Hash-verified full history, opened only after freeze checks.
    features = build_features(matches)
    test = matches.loc[matches.Season.isin(TEST_SEASONS)].sort_values(KEY)
    X_test = features.loc[test.index]
    forecasts = {}
    for family, pipeline in bundle["pipelines"].items():
        imputer_before = pipeline.named_steps["imputer"].statistics_.copy()
        scaler_before = pipeline.named_steps["scaler"].mean_.copy() if "scaler" in pipeline.named_steps else None
        forecasts[decision["family_candidates"][family]] = predict_hda(pipeline, X_test, metadata["config"]["threads"])
        np.testing.assert_array_equal(pipeline.named_steps["imputer"].statistics_, imputer_before)
        if scaler_before is not None:
            np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, scaler_before)
    frequency = np.asarray(bundle["training_frequency"])
    forecasts["training_frequency"] = np.tile(frequency, (len(test), 1))
    bookmaker = bookmaker_probabilities(test)
    results, classification, coverage = evaluation_tables(test, forecasts, bookmaker)
    results.to_csv(reports / "test_results.csv", index=False)
    classification.to_csv(reports / "test_classification.csv", index=False)
    coverage.to_csv(reports / "test_coverage.csv", index=False)
    winner_probs = forecasts[decision["winner_id"]]
    bins = calibration_bins(test.FTR.to_numpy(), winner_probs)
    bins.to_csv(reports / "calibration_bins.csv", index=False)
    plot_calibration(bins, reports / "calibration.png", decision["winner_id"])
    groups = error_groups(test, X_test, winner_probs)
    groups.to_csv(reports / "test_error_groups.csv", index=False)
    exports = []
    for forecast, probabilities in {**forecasts, "bookmaker": bookmaker.to_numpy()}.items():
        frame = test[KEY + ["FTR"]].copy()
        frame["fixture_id"], frame["split"], frame["forecast"] = fixture_ids(test), "test", forecast
        frame["odds_valid"] = bookmaker.notna().all(axis=1)
        frame[PROBABILITY_COLUMNS] = probabilities
        exports.append(frame)
    pd.concat(exports, ignore_index=True).to_csv(reports / "test_predictions.csv", index=False)
    errors = test[KEY + ["FTR", "FTHG", "FTAG"]].copy()
    errors["fixture_id"] = fixture_ids(test)
    errors[PROBABILITY_COLUMNS] = winner_probs
    errors["predicted_result"] = np.asarray(CLASS_ORDER)[winner_probs.argmax(axis=1)]
    actual_indices = np.array([CLASS_ORDER.index(label) for label in test.FTR])
    errors["log_loss"] = -np.log(np.maximum(winner_probs[np.arange(len(test)), actual_indices], LOG_LOSS_EPSILON))
    errors[["home_history_count_5", "away_history_count_5"]] = X_test[["home_history_count_5", "away_history_count_5"]]
    errors.sort_values("log_loss", ascending=False).head(5).to_csv(reports / "test_error_examples.csv", index=False)
    first = test.iloc[0]
    example = {"fixture": {**first[KEY].to_dict(), "Date": first.Date.date().isoformat()},
               "winner_id": decision["winner_id"], "probabilities_h_d_a": dict(zip(CLASS_ORDER, winner_probs[0].tolist())),
               "actual_result": first.FTR, "predicted_result": CLASS_ORDER[int(winner_probs[0].argmax())],
               "log_loss": float(errors.iloc[0].log_loss),
               "features": {name: None if pd.isna(value) else float(value) for name, value in X_test.iloc[0].items()}}
    (reports / "test_example.json").write_text(json.dumps(example, indent=2, allow_nan=False) + "\n")
    if file_hash(reports / "selected_pipelines.joblib") != decision["artifact_sha256"] or file_hash(reports / "winner_decision.json") != summary["winner_decision_sha256"]:
        raise ValueError("Frozen pipeline or selection was modified during evaluation")
    names = ["test_results.csv", "test_classification.csv", "test_coverage.csv", "test_predictions.csv", "calibration_bins.csv",
             "calibration.png", "calibration.pdf", "test_error_groups.csv", "test_error_examples.csv", "test_example.json"]
    summary.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "test_matches": len(test),
                    "coverage": coverage.to_dict("records"), "output_sha256": {name: file_hash(reports / name) for name in names},
                    "test_evaluated": True, "winner_preserved": True})
    marker.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return display_results(reports)


if __name__ == "__main__":
    run_evaluation()
