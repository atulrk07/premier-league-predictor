"""Protect saved-result integrity, fixture alignment, and dashboard interactions."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="Install requirements-dashboard.txt for dashboard checks")
from streamlit.testing.v1 import AppTest

import src.dashboard_data as dashboard
from src.data import ROOT
from src.metrics import CLASS_ORDER, multiclass_log_loss


def original_hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "reports").iterdir() if path.is_file()}


def test_snapshot_needs_no_raw_data_or_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical display must not build features, load raw data, or predict")
    for name in ("load_data", "build_features", "predict_hda", "load_frozen"):
        monkeypatch.setattr(dashboard, name, forbidden)
    before = original_hashes()
    data = dashboard.load_snapshot()
    assert len(data["matches"]) == 1140
    assert data["matches"].groupby("split").size().to_dict() == {"test": 760, "validation": 380}
    assert original_hashes() == before


def test_historical_fixture_and_saved_probabilities_agree():
    data = dashboard.load_snapshot()
    example = json.loads((ROOT / "reports/test_example.json").read_text())
    fixture = data["matches"].loc[data["matches"].fixture_id.eq("2024/25|2024-08-16|Man United|Fulham")].iloc[0]
    expected = [example["probabilities_h_d_a"][outcome] for outcome in CLASS_ORDER]
    np.testing.assert_allclose(fixture[dashboard.PROBABILITIES].to_numpy(dtype=float), expected, atol=1e-12)
    assert sum(expected) == pytest.approx(1)
    assert fixture.FTHG == 1 and fixture.FTAG == 0 and fixture.FTR == "H"
    for name, value in example["features"].items():
        assert fixture[name] == pytest.approx(value)
    assert dashboard.fixture_scores(fixture)[0] == pytest.approx(example["log_loss"])
    saved = pd.read_csv(ROOT / "reports/test_predictions.csv")
    book = saved.loc[saved.fixture_id.eq(fixture.fixture_id) & saved.forecast.eq("bookmaker")].iloc[0]
    np.testing.assert_allclose(fixture[dashboard.BOOKMAKER].to_numpy(dtype=float), book[dashboard.PROBABILITIES].to_numpy(dtype=float))


def test_baseline_improvement_sign_and_denominators():
    data = dashboard.load_snapshot()
    results = data["results"]
    matched = results.loc[results.season.eq("combined") & results.subset.eq("matched_bookmaker")]
    improvements = dashboard.baseline_improvements(matched, "lr_c0.1")
    assert improvements.iloc[0]["Log loss reduction"] == pytest.approx(1.0821814716449862 - 1.036151924114111)
    assert improvements.iloc[1]["Log loss reduction"] < 0
    assert improvements.iloc[1]["Accuracy gain (percentage points)"] < 0
    unequal = matched.copy()
    unequal.loc[unequal.forecast.eq("bookmaker"), "matches"] = 759
    with pytest.raises(ValueError, match="same season, subset, and match count"):
        dashboard.baseline_improvements(unequal, "lr_c0.1")


def test_outcome_analysis_matches_original_reports():
    data = dashboard.load_snapshot()
    frame = data["matches"].loc[data["matches"].split.eq("test")]
    bins, matrix, performance = dashboard.outcome_analysis(frame)
    np.testing.assert_allclose(bins.select_dtypes(include="number"), data["calibration"].select_dtypes(include="number"), equal_nan=True)
    assert matrix.to_numpy().sum() == 760
    draw = performance.loc[performance.outcome.eq("D")].iloc[0]
    assert draw.support == 197 and draw.predicted_count == 1 and draw.true_positive == 1
    results = data["results"]
    score = results.loc[results.season.eq("combined") & results.subset.eq("all_test") & results.forecast.eq("lr_c0.1")].iloc[0]
    assert multiclass_log_loss(frame.FTR, frame[dashboard.PROBABILITIES]) == pytest.approx(score.log_loss)


def test_snapshot_hash_mismatch_is_reported(tmp_path):
    folder = tmp_path / "data/dashboard"
    folder.mkdir(parents=True)
    manifest = json.loads((ROOT / dashboard.MANIFEST).read_text())
    (folder / "manifest.json").write_text(json.dumps(manifest))
    for name in dashboard.INPUTS:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(ROOT / name)
    (folder / "matches.csv").write_text("changed snapshot")
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        dashboard.load_snapshot(tmp_path)


def test_dashboard_smoke_and_filters_preserve_artifacts(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Widget interactions must never load raw data or run inference")
    for name in ("load_data", "build_features", "predict_hda", "load_frozen"):
        monkeypatch.setattr(dashboard, name, forbidden)
    before = original_hashes()
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Results overview", "Historical match explorer", "Model analysis", "Methodology"]
    assert app.metric[0].value == "1.036152"
    assert app.selectbox(key="fixture").value == "2024/25|2024-08-16|Man United|Fulham"
    assert [item.value for item in app.metric[3:6]] == ["41.65%", "21.87%", "36.47%"]
    app.selectbox(key="results_season").select("2025/26").run()
    assert not app.exception
    assert app.metric[0].value == "1.057667"
    app.selectbox(key="match_season").select("2023/24").run()
    app.selectbox(key="team").select("Burnley").run()
    assert not app.exception
    assert app.selectbox(key="fixture").value == "2023/24|2023-08-11|Burnley|Man City"
    assert [item.value for item in app.metric[3:6]] == ["11.95%", "14.70%", "73.35%"]
    assert original_hashes() == before


def test_dashboard_missing_snapshot_has_helpful_error(monkeypatch):
    def missing(*args):
        raise FileNotFoundError("dashboard matches snapshot is missing")
    monkeypatch.setattr(dashboard, "snapshot_signature", missing)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    assert "snapshot is missing" in app.error[0].value
    assert "python -m src.dashboard_data" in app.code[0].value
