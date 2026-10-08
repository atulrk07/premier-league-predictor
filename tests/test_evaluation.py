"""Synthetic evaluation checks do not open real final-test outcomes."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import calibration_bins, classification_summary, error_groups, evaluation_tables, file_hash, load_frozen, run_evaluation
from src.experiment import run_experiment


def test_matched_comparisons_use_identical_fixture_subset():
    matches = pd.DataFrame({"Season": ["2024/25", "2024/25", "2025/26"], "FTR": ["H", "D", "A"]}, index=[4, 8, 12])
    forecast = np.tile([0.5, 0.3, 0.2], (3, 1))
    bookmaker = pd.DataFrame([[0.6, 0.2, 0.2], [np.nan] * 3, [0.2, 0.2, 0.6]], columns=["H", "D", "A"], index=matches.index)
    results, classes, coverage = evaluation_tables(matches, {"fixed": forecast}, bookmaker)
    combined = results.loc[results.season.eq("combined")]
    all_rows = combined.loc[combined.subset.eq("all_test")].iloc[0]
    assert all_rows.matches == 3
    assert all_rows.log_loss == pytest.approx((-np.log(0.5) - np.log(0.3) - np.log(0.2)) / 3)
    matched = combined.loc[combined.subset.eq("matched_bookmaker")].set_index("forecast")
    assert matched.matches.tolist() == [2, 2]
    assert matched.loc["fixed", "log_loss"] == pytest.approx((-np.log(0.5) - np.log(0.2)) / 2)
    assert matched.loc["bookmaker", "log_loss"] == pytest.approx(-np.log(0.6))
    assert coverage.loc[coverage.season.eq("combined"), "coverage"].iloc[0] == pytest.approx(2 / 3)
    assert classes.loc[(classes.season == "combined") & (classes.subset == "matched_bookmaker") & (classes.outcome == "D"), "support"].eq(0).all()
    with pytest.raises(ValueError, match="align"):
        evaluation_tables(matches, {"fixed": forecast}, bookmaker.iloc[::-1])


def test_zero_bookmaker_coverage_keeps_model_rows():
    matches = pd.DataFrame({"Season": ["2024/25"], "FTR": ["H"]})
    bookmaker = pd.DataFrame([[np.nan] * 3], columns=["H", "D", "A"])
    results, _, coverage = evaluation_tables(matches, {"model": np.array([[1, 0, 0]])}, bookmaker)
    row = results.loc[(results.season == "combined") & (results.subset == "all_test")].iloc[0]
    assert row.matches == 1 and row.log_loss == 0
    assert coverage.loc[coverage.season.eq("combined"), "coverage"].iloc[0] == 0
    assert results.loc[results.subset.eq("matched_bookmaker"), "matches"].eq(0).all()


def test_calibration_counts_boundaries_and_observed_frequency():
    probabilities = np.array([[0, 1, 0], [0.2, 0.8, 0], [0.21, 0.79, 0], [1, 0, 0]])
    bins = calibration_bins(["H", "D", "H", "H"], probabilities)
    assert len(bins) == 15
    assert bins.groupby("outcome")["count"].sum().eq(4).all()
    home = bins.loc[bins.outcome.eq("H")].set_index("bin")
    assert home.loc[1, "count"] == 2
    assert home.loc[1, "mean_probability"] == pytest.approx(0.1)
    assert home.loc[1, "observed_frequency"] == 0.5
    assert home.loc[2, "count"] == 1 and home.loc[2, "observed_frequency"] == 1
    assert home.loc[5, "count"] == 1 and home.loc[5, "observed_frequency"] == 1
    assert pd.isna(home.loc[3, "observed_frequency"])


def test_draw_precision_recall_and_support():
    probabilities = np.eye(3)[[0, 0, 1, 2]]
    summary = pd.DataFrame(classification_summary(["H", "D", "D", "A"], probabilities)).set_index("outcome")
    assert summary.loc["D", "precision"] == 1
    assert summary.loc["D", "recall"] == 0.5
    assert summary.loc["D", "support"] == 2
    assert summary.loc["D", "predicted_count"] == 1


def test_low_history_threshold_and_group_denominators():
    matches = pd.DataFrame({"Season": ["2024/25"] * 3, "FTR": ["H", "D", "A"]})
    features = pd.DataFrame({"home_history_count_5": [0, 4, 5], "away_history_count_5": [5, 5, 5]})
    probabilities = np.eye(3)
    rows = error_groups(matches, features, probabilities)
    combined = rows.loc[rows.season.eq("combined")].set_index("group")
    assert combined.loc["low_history_either_team_lt5", "matches"] == 2
    assert combined.loc["full_history_both_5", "matches"] == 1
    assert combined.loc["zero_history_either_team", "matches"] == 1
    assert combined.loc["actual_D", "matches"] == 1
    assert combined.loc["all", "log_loss"] == 0


def test_frozen_artifact_mismatch_fails_before_test_access(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "winner_decision.json").write_text(json.dumps({"artifact_sha256": "0" * 64}))
    (reports / "experiment_config.json").write_text("{}")
    (reports / "selected_pipelines.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_frozen(tmp_path)


def test_training_cannot_overwrite_freeze_after_test_started(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "test_evaluation.json").write_text('{"status": "started"}')
    monkeypatch.setattr("src.experiment.ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="do not refit"):
        run_experiment()


def test_completed_evaluation_uses_cache_without_prediction_or_test_loading(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    decision_path = reports / "winner_decision.json"
    decision_path.write_text("{}")
    marker = {"status": "complete", "frozen_artifact_sha256": "frozen",
              "winner_decision_sha256": file_hash(decision_path), "source_sha256": {"2024/25": "snapshot"},
              "output_sha256": {}}
    (reports / "test_evaluation.json").write_text(json.dumps(marker))
    monkeypatch.setattr("src.evaluation.ROOT", tmp_path)
    monkeypatch.setattr("src.evaluation.load_frozen", lambda: ({}, {"artifact_sha256": "frozen"}, {}, {"2024/25": {"sha256": "snapshot"}}))

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed evaluation must not load test outcomes or recompute predictions")

    monkeypatch.setattr("src.evaluation.load_data", forbidden)
    monkeypatch.setattr("src.evaluation.predict_hda", forbidden)
    monkeypatch.setattr("src.evaluation.display_results", lambda path: {"cached": True})
    assert run_evaluation() == {"cached": True}
