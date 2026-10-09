"""Read-only historical dashboard data; prepare its separate snapshot explicitly.

python -m src.dashboard_data builds display metadata from the recorded archives.
The app itself only reads that snapshot and the original saved reports.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import KEY, ROOT, SEASONS, load_data
from src.evaluation import calibration_bins, classification_summary, file_hash, load_frozen
from src.experiment import predict_hda
from src.features import FEATURE_COLUMNS, TEAM_FEATURES, build_features
from src.metrics import CLASS_ORDER, bookmaker_probabilities, multiclass_log_loss, ranked_probability_score

SNAPSHOT = "data/dashboard/matches.csv"
MANIFEST = "data/dashboard/manifest.json"
PROBABILITIES = [f"p_{outcome}" for outcome in CLASS_ORDER]
BOOKMAKER = [f"book_p_{outcome}" for outcome in CLASS_ORDER]
INPUTS = (
    "config/experiment.json", "data/sources.json", "reports/experiment_config.json",
    "reports/selected_pipelines.joblib", "reports/winner_decision.json", "reports/test_evaluation.json",
    "reports/test_predictions.csv", "reports/validation_predictions.csv", "reports/test_results.csv",
    "reports/test_coverage.csv", "reports/test_classification.csv", "reports/calibration_bins.csv",
    "reports/validation_results.csv", "reports/frequency_validation.json",
)
FEATURE_LABELS = dict(zip(TEAM_FEATURES, (
    "Mean points · last 5", "Mean points · last 10", "Mean goals scored · last 5",
    "Mean goals conceded · last 5", "Matches available · last 5",
    "Matches available · last 10", "Rest days (capped at 60)",
)))
FORECAST_LABELS = {
    "lr_c0.1": "Logistic regression · C=0.1 (selected)",
    "rf_d8_leaf10": "Random Forest · depth 8 / leaf 10",
    "xgb_d2_n200_lr0.05": "XGBoost · depth 2 / 200 rounds / rate 0.05",
    "training_frequency": "Training frequency", "bookmaker": "Normalized Bet365 odds",
}


def prepare_snapshot(root=ROOT):
    """Build features once, join saved probabilities, verify frozen inference; never fit."""
    root = Path(root)
    bundle, decision, metadata, _ = load_frozen(root)
    evaluation = json.loads((root / "reports/test_evaluation.json").read_text())
    if evaluation["status"] != "complete" or evaluation["winner_id"] != decision["winner_id"]:
        raise ValueError("Complete the original frozen evaluation before preparing the dashboard")
    for name, digest in evaluation["output_sha256"].items():
        if file_hash(root / "reports" / name) != digest:
            raise ValueError(f"Original evaluation output changed: {name}")
    original_hashes = {name: file_hash(root / name) for name in INPUTS}
    matches, _ = load_data(root=root)
    features = build_features(matches)
    held_out = matches.loc[matches.Season.isin(SEASONS[9:])]
    validation = pd.read_csv(root / "reports/validation_predictions.csv").rename(columns={"candidate_id": "forecast"})
    test = pd.read_csv(root / "reports/test_predictions.csv")
    saved = pd.concat([validation, test], ignore_index=True)
    saved = saved.loc[saved.forecast.eq(decision["winner_id"])].copy()
    saved["Date"] = pd.to_datetime(saved.Date, format="%Y-%m-%d")
    frame = held_out[KEY + ["FTHG", "FTAG", "FTR"]].merge(
        saved[KEY + ["FTR", "split"] + PROBABILITIES], on=KEY, validate="one_to_one",
        how="left", suffixes=("_raw", ""),
    )
    if frame[PROBABILITIES].isna().any().any() or not frame.FTR_raw.eq(frame.FTR).all():
        raise ValueError("Saved forecasts do not align with the recorded match results")
    frame = frame.drop(columns="FTR_raw")
    inputs = pd.concat([held_out[KEY], features.loc[held_out.index]], axis=1)
    frame = frame.merge(inputs, on=KEY, validate="one_to_one")
    book = bookmaker_probabilities(held_out).set_axis(BOOKMAKER, axis=1)
    frame = frame.merge(pd.concat([held_out[KEY], book], axis=1), on=KEY, validate="one_to_one")
    saved_book = test.loc[test.forecast.eq("bookmaker"), KEY + PROBABILITIES].copy()
    saved_book["Date"] = pd.to_datetime(saved_book.Date, format="%Y-%m-%d")
    saved_book = saved_book.rename(columns=dict(zip(PROBABILITIES, BOOKMAKER)))
    comparison = frame.loc[frame.split.eq("test"), KEY + BOOKMAKER].merge(
        saved_book, on=KEY, validate="one_to_one", how="left", suffixes=("", "_saved"),
    )
    np.testing.assert_allclose(comparison[BOOKMAKER], comparison[[f"{name}_saved" for name in BOOKMAKER]],
                               atol=1e-10, rtol=0, equal_nan=True)
    frame["fixture_id"] = frame[KEY].assign(Date=frame.Date.dt.strftime("%Y-%m-%d")).astype(str).agg("|".join, axis=1)
    pipeline = bundle["pipelines"][decision["winner_family"]]
    reproduced = predict_hda(pipeline, frame[list(FEATURE_COLUMNS)], metadata["config"]["threads"])
    np.testing.assert_allclose(reproduced, frame[PROBABILITIES], atol=1e-10, rtol=0)
    if any(file_hash(root / name) != digest for name, digest in original_hashes.items()):
        raise ValueError("Original artifacts changed while preparing the dashboard")
    destination = root / SNAPSHOT
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(KEY).to_csv(destination, index=False, date_format="%Y-%m-%d")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
        "winner_id": decision["winner_id"], "class_order": list(CLASS_ORDER),
        "feature_columns": list(FEATURE_COLUMNS), "snapshot_sha256": file_hash(destination),
        "input_sha256": original_hashes, "source_sha256": evaluation["source_sha256"],
        "verification": "Frozen pipeline reproduces saved H/D/A predictions within 1e-10; no fitting",
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def snapshot_signature(root=ROOT):
    """Cheap file statistics invalidate Streamlit's cache when inputs change."""
    root = Path(root)
    return tuple((name, (root / name).stat().st_mtime_ns, (root / name).stat().st_size)
                 for name in (MANIFEST, SNAPSHOT, *INPUTS))


def load_snapshot(root=ROOT):
    """Verify display data and original inputs without opening any raw archive."""
    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest["class_order"] != list(CLASS_ORDER) or manifest["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("Dashboard snapshot uses different classes or feature definitions")
    if set(manifest["input_sha256"]) != set(INPUTS):
        raise ValueError("Dashboard snapshot is missing original artifact provenance")
    for name, digest in manifest["input_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Dashboard snapshot no longer matches {name}; review the artifact change")
    if file_hash(root / SNAPSHOT) != manifest["snapshot_sha256"]:
        raise ValueError("Dashboard match snapshot hash mismatch")
    frame = pd.read_csv(root / SNAPSHOT, parse_dates=["Date"])
    if len(frame) != manifest["rows"] or frame.fixture_id.duplicated().any():
        raise ValueError("Dashboard fixture counts or identifiers are invalid")
    multiclass_log_loss(frame.FTR, frame[PROBABILITIES])
    valid_book = frame[BOOKMAKER].notna().all(axis=1)
    if frame[BOOKMAKER].isna().any(axis=1).ne(frame[BOOKMAKER].isna().all(axis=1)).any():
        raise ValueError("Bookmaker probability triplets must be wholly present or wholly missing")
    if valid_book.any():
        multiclass_log_loss(frame.loc[valid_book, "FTR"], frame.loc[valid_book, BOOKMAKER])
    data = {"matches": frame, "manifest": manifest}
    for key, name in {
        "results": "test_results.csv", "coverage": "test_coverage.csv", "classification": "test_classification.csv",
        "calibration": "calibration_bins.csv", "validation": "validation_results.csv",
    }.items():
        data[key] = pd.read_csv(root / "reports" / name)
    for key, name in {
        "decision": "winner_decision.json", "metadata": "experiment_config.json",
        "frequency": "frequency_validation.json", "evaluation": "test_evaluation.json",
    }.items():
        data[key] = json.loads((root / "reports" / name).read_text())
    return data


def baseline_improvements(results, winner_id):
    """Compare saved scores on one identical season/subset; positive means improvement."""
    if results.season.nunique() != 1 or results.subset.nunique() != 1 or results.matches.nunique() != 1:
        raise ValueError("Baseline comparisons require the same season, subset, and match count")
    winner = results.loc[results.forecast.eq(winner_id)].iloc[0]
    rows = []
    for baseline in results.loc[results.forecast.isin(["training_frequency", "bookmaker"])].itertuples():
        reduction = baseline.log_loss - winner.log_loss
        rows.append({"Baseline": FORECAST_LABELS[baseline.forecast], "Log loss reduction": reduction,
                     "Relative log loss reduction (%)": 100 * reduction / baseline.log_loss if baseline.log_loss else np.nan,
                     "RPS reduction": baseline.rps - winner.rps,
                     "Accuracy gain (percentage points)": 100 * (winner.accuracy - baseline.accuracy)})
    return pd.DataFrame(rows)


def outcome_analysis(frame):
    """Descriptive statistics from saved forecasts; reuse the original metric/calibration code."""
    probabilities = frame[PROBABILITIES].to_numpy()
    predicted = np.asarray(CLASS_ORDER)[probabilities.argmax(axis=1)]
    matrix = pd.crosstab(pd.Categorical(frame.FTR, categories=CLASS_ORDER),
                         pd.Categorical(predicted, categories=CLASS_ORDER), dropna=False)
    matrix.index = [f"Actual {outcome}" for outcome in CLASS_ORDER]
    matrix.columns = [f"Predicted {outcome}" for outcome in CLASS_ORDER]
    return calibration_bins(frame.FTR, probabilities), matrix, pd.DataFrame(classification_summary(frame.FTR, probabilities))


def fixture_scores(row):
    probabilities = np.array([[row[name] for name in PROBABILITIES]])
    return multiclass_log_loss([row.FTR], probabilities), ranked_probability_score([row.FTR], probabilities)


if __name__ == "__main__":
    prepared = prepare_snapshot()
    print(f"Prepared {prepared['rows']} historical fixtures in {SNAPSHOT}; original artifacts unchanged.")
