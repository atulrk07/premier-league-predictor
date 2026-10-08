"""Synthetic checks for chronology, preprocessing fit boundaries, and selection."""

import numpy as np
import pandas as pd
import pytest

from src.data import SEASONS
from src.experiment import chronological_split, fit_training, load_config, load_development_data, make_pipeline, predict_hda, select_models
from src.features import FEATURE_COLUMNS


def season_fixtures():
    return pd.DataFrame({
        "Season": SEASONS[:10], "Date": pd.to_datetime([f"{year}-08-01" for year in range(2014, 2024)]),
        "HomeTeam": ["Home"] * 10, "AwayTeam": ["Away"] * 10,
    })


def test_split_is_chronological_and_rejects_test_seasons():
    frame = season_fixtures()
    train, validation = chronological_split(frame)
    assert train.Season.tolist() == SEASONS[:9] and validation.Season.tolist() == ["2023/24"]
    assert train.Date.max() < validation.Date.min()
    for test_season in SEASONS[10:]:
        extra = frame.iloc[[-1]].assign(Season=test_season)
        with pytest.raises(ValueError, match="refuses final-test"):
            chronological_split(pd.concat([frame, extra], ignore_index=True))
    reversed_dates = frame.copy()
    reversed_dates.loc[8, "Date"] = pd.Timestamp("2023-08-02")
    with pytest.raises(ValueError, match="precede"):
        chronological_split(reversed_dates)


def test_development_loader_requests_no_final_test_files(monkeypatch):
    requests = []

    def loader(*, seasons):
        requests.append(seasons)
        return season_fixtures(), []

    monkeypatch.setattr("src.experiment.load_data", loader)
    _, train, validation = load_development_data()
    assert requests == [SEASONS[:10]]
    assert not set(SEASONS[10:]).intersection(requests[0])
    assert len(train) == 9 and len(validation) == 1


def test_imputation_scaling_and_prediction_use_training_statistics_only():
    config = load_config()
    X = pd.DataFrame(np.tile(np.array([1, 2, 3])[:, None], (1, 14)), columns=FEATURE_COLUMNS, dtype=float)
    X.loc[1, FEATURE_COLUMNS[0]] = np.nan
    labels = pd.DataFrame({"Season": ["2022/23"] * 3, "FTR": ["H", "D", "A"]})
    pipeline = fit_training(make_pipeline(config["candidates"][0], config), X, labels, 1)
    imputer, scaler = pipeline.named_steps["imputer"], pipeline.named_steps["scaler"]
    np.testing.assert_allclose(imputer.statistics_, [2] * 14)
    assert imputer.indicator_.features_.tolist() == [0]
    np.testing.assert_allclose(scaler.mean_, [2] * 14 + [1 / 3])
    assert pipeline.named_steps["model"].n_features_in_ == 15
    medians_before, means_before = imputer.statistics_.copy(), scaler.mean_.copy()
    X_val = X.iloc[:2].copy()
    X_val.iloc[0] = 1000
    X_val.iloc[1, 0] = np.nan
    probabilities = predict_hda(pipeline, X_val)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(imputer.statistics_, medians_before)
    np.testing.assert_array_equal(scaler.mean_, means_before)
    # Extra outcome/odds columns are dropped by the pipeline's own allowlist.
    np.testing.assert_allclose(probabilities, predict_hda(pipeline, X_val.assign(FTHG=99, B365H=1.1)))
    for season in ["2023/24", "2024/25"]:
        with pytest.raises(ValueError, match="training seasons only"):
            fit_training(pipeline, X, labels.assign(Season=season), 1)
    with pytest.raises(ValueError, match="align"):
        fit_training(pipeline, X.iloc[::-1], labels, 1)


def test_prd_eight_configurations_and_scaling_policy():
    config = load_config()
    candidates = config["candidates"]
    assert len(candidates) == len({candidate["id"] for candidate in candidates}) == 8
    assert [c["params"]["C"] for c in candidates if c["family"] == "logistic_regression"] == [0.1, 1.0]
    assert [(c["params"]["max_depth"], c["params"]["min_samples_leaf"]) for c in candidates if c["family"] == "random_forest"] == [(5, 10), (8, 10), (8, 20)]
    assert [(c["params"]["max_depth"], c["params"]["n_estimators"], c["params"]["learning_rate"]) for c in candidates if c["family"] == "xgboost"] == [(2, 200, 0.05), (3, 200, 0.05), (3, 400, 0.03)]
    for candidate in candidates:
        pipeline = make_pipeline(candidate, config)
        assert ("scaler" in pipeline.named_steps) == (candidate["family"] == "logistic_regression")
        assert pipeline.named_steps["model"].random_state == 42
        assert pipeline.named_steps["imputer"].strategy == "median"
        assert pipeline.named_steps["imputer"].add_indicator is True


@pytest.mark.parametrize("lr_loss,rf_loss,xgb_loss,expected", [
    (0.954, 0.950, 0.951, "lr_best"),
    (0.955, 0.950, 0.951, "lr_best"),  # exact 0.005 boundary
    (0.956, 0.950, 0.951, "rf_best"),
    (0.956, 0.954, 0.950, "rf_best"),
    (0.960, 0.958, 0.950, "xgb_best"),
])
def test_selection_uses_family_minima_and_simplicity_tolerance(lr_loss, rf_loss, xgb_loss, expected):
    results = pd.DataFrame({
        "candidate_id": ["lr_worse", "lr_best", "rf_best", "xgb_best"],
        "family": ["logistic_regression", "logistic_regression", "random_forest", "xgboost"],
        "validation_log_loss": [lr_loss + 0.02, lr_loss, rf_loss, xgb_loss],
    })
    decision = select_models(results)
    assert decision["winner_id"] == expected
    assert decision["family_candidates"]["logistic_regression"] == "lr_best"
    assert decision["refit_on_validation"] is False and decision["test_evaluated"] is False


def test_probability_mapping_handles_estimator_order_and_float_rounding():
    class Estimator:
        classes_ = np.array([2, 0, 1])

        def predict_proba(self, X):
            return np.tile([0.2, 0.5, 0.30000001], (len(X), 1))

    probabilities = predict_hda(Estimator(), [1])
    np.testing.assert_allclose(probabilities, [[0.5, 0.3, 0.2]], atol=1e-8)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-10)
