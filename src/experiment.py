"""Milestone 4: eight training fits, validation selection, and frozen artifacts.

Run `python -m src.experiment`. Only 2014/15–2023/24 CSVs are loaded.
No final-test evaluation command is implemented in this milestone.
"""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.data import KEY, ROOT, SEASONS, load_data
from src.features import FEATURE_COLUMNS, HISTORY_DAYS, REST_CAP_DAYS, build_features
from src.metrics import CLASS_ORDER, LOG_LOSS_EPSILON, multiclass_log_loss, ranked_probability_score, reorder_probabilities, training_frequency

CONFIG_PATH = ROOT / "config/experiment.json"


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def chronological_split(matches):
    """Reject test rows and require both complete season sets before fitting."""
    allowed = SEASONS[:10]
    if not matches.Season.isin(allowed).all():
        raise ValueError("Milestone 4 refuses final-test or unknown seasons")
    if set(matches.Season.unique()) != set(allowed):
        raise ValueError("Expected every training season and the 2023/24 validation season")
    if matches.duplicated(KEY).any() or not matches.index.is_unique:
        raise ValueError("Fixture keys and row indices must be unique")
    train = matches.loc[matches.Season.isin(SEASONS[:9])].copy()
    validation = matches.loc[matches.Season.eq("2023/24")].copy()
    if train.Date.isna().any() or validation.Date.isna().any() or train.Date.max() >= validation.Date.min():
        raise ValueError("Every training date must precede every validation date")
    return train, validation


def load_development_data():
    """Open only training/validation CSVs; never open either final-test CSV."""
    matches, _ = load_data(seasons=SEASONS[:10])
    train, validation = chronological_split(matches)
    return matches, train, validation


def make_pipeline(candidate, config):
    family = candidate["family"]
    params = {**config["fixed_parameters"][family], **candidate["params"], "random_state": config["seed"]}
    if family == "logistic_regression":
        classifier = LogisticRegression(**params)
    elif family == "random_forest":
        classifier = RandomForestClassifier(**params, n_jobs=config["threads"])
    elif family == "xgboost":
        classifier = XGBClassifier(**params, n_jobs=config["threads"])
    else:
        raise ValueError(f"Unknown family: {family}")
    steps = [
        ("features", ColumnTransformer([("allowed", "passthrough", list(FEATURE_COLUMNS))], remainder="drop", verbose_feature_names_out=False)),
        ("imputer", SimpleImputer(**{key: config["preprocessing"][key] for key in ["strategy", "add_indicator", "keep_empty_features"]})),
    ]
    if family == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("model", classifier)])


def fit_training(pipeline, X, train_matches, threads):
    """All learned preprocessing and model parameters see training rows only."""
    if not train_matches.Season.isin(SEASONS[:9]).all() or len(train_matches) == 0:
        raise ValueError("Fitting accepts training seasons only")
    if not X.index.equals(train_matches.index) or tuple(X.columns) != FEATURE_COLUMNS:
        raise ValueError("Training features must align and use the exact feature allowlist")
    if set(train_matches.FTR.unique()) != set(CLASS_ORDER):
        raise ValueError("Training labels must contain all three H D A outcomes")
    labels = train_matches.FTR.map(dict(zip(CLASS_ORDER, range(3)))).to_numpy()
    with threadpool_limits(limits=threads), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, labels)
    return pipeline


def predict_hda(pipeline, X, threads=1):
    """Reuse fitted preprocessing; map encoded estimator classes back to H/D/A."""
    with threadpool_limits(limits=threads):
        probabilities = np.asarray(pipeline.predict_proba(X), dtype=float)
    classes = np.asarray(pipeline.classes_)
    if set(classes.tolist()) != {0, 1, 2}:
        raise ValueError("Expected estimator classes encoded as H=0, D=1, A=2")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Estimator probabilities must be finite and between 0 and 1")
    totals = probabilities.sum(axis=1)
    if not np.allclose(totals, 1, atol=1e-6, rtol=0):
        raise ValueError("Estimator probability rows must sum to 1")
    # XGBoost's float32 outputs can drift slightly from 1; correct only rounding.
    probabilities = probabilities / totals[:, None]
    return reorder_probabilities(probabilities, [CLASS_ORDER[int(value)] for value in classes])


def score_predictions(train_labels, validation_labels, train_probs, validation_probs):
    return {
        "train_log_loss": multiclass_log_loss(train_labels, train_probs),
        "validation_log_loss": multiclass_log_loss(validation_labels, validation_probs),
        "validation_rps": ranked_probability_score(validation_labels, validation_probs),
        "validation_accuracy": float(np.mean(np.asarray(CLASS_ORDER)[validation_probs.argmax(axis=1)] == validation_labels)),
    }


def select_models(results, tolerance=0.005, preference=("logistic_regression", "random_forest", "xgboost")):
    """Family minima first; simplicity preference within tolerance of best minimum."""
    if set(results.family) != set(preference) or results.candidate_id.duplicated().any():
        raise ValueError("Need distinct candidates covering exactly the three model families")
    if not np.isfinite(results.validation_log_loss).all():
        raise ValueError("Selection requires finite validation log losses")
    family_best = results.sort_values(["validation_log_loss", "candidate_id"]).groupby("family", sort=False).head(1)
    lowest = float(family_best.validation_log_loss.min())
    eligible = family_best[family_best.validation_log_loss <= lowest + tolerance + 1e-12]
    chosen = eligible.assign(priority=eligible.family.map({name: i for i, name in enumerate(preference)})).sort_values("priority").iloc[0]
    return {
        "family_candidates": dict(zip(family_best.family, family_best.candidate_id)),
        "lowest_validation_log_loss": lowest,
        "eligible_within_tolerance": eligible.candidate_id.tolist(),
        "winner_id": chosen.candidate_id, "winner_family": chosen.family,
        "winner_validation_log_loss": float(chosen.validation_log_loss),
        "tolerance": tolerance, "family_preference": list(preference),
        "rule": "Minimum validation log loss per family; within 0.005 of global minimum prefer logistic regression, Random Forest, then XGBoost.",
        "refit_on_validation": False, "test_evaluated": False,
    }


def run_experiment():
    if (ROOT / "reports/test_evaluation.json").exists():
        raise RuntimeError("Final-test evaluation has started. Preserve the frozen models; do not refit this experiment after test access.")
    config = load_config()
    if config["train_seasons"] != SEASONS[:9] or config["validation_seasons"] != ["2023/24"] or config["test_seasons"] != SEASONS[10:]:
        raise ValueError("Config must retain the PRD chronological split")
    if config["feature_columns"] != list(FEATURE_COLUMNS) or len(config["candidates"]) != 8:
        raise ValueError("Config must retain 14 raw features and the eight-candidate budget")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    # Record the experiment specification before any fits or validation scoring.
    sources = json.loads((ROOT / "data/sources.json").read_text())
    metadata = {
        "config": config, "class_mapping": {label: i for i, label in enumerate(CLASS_ORDER)},
        "history_days": HISTORY_DAYS, "rest_cap_days": REST_CAP_DAYS,
        "log_loss_epsilon": LOG_LOSS_EPSILON, "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "xgboost", "joblib", "threadpoolctl"]},
        "source_sha256": {season: sources[season]["sha256"] for season in SEASONS[:10]},
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": "Milestone 4 loads only 2014/15–2023/24 raw files. Test CSVs are not opened.",
    }
    (reports / "experiment_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    matches, train, validation = load_development_data()
    # Build once with continuous history; same-day batching keeps labels date-safe.
    features = build_features(matches)
    X_train, X_validation = features.loc[train.index], features.loc[validation.index]
    fitted, predictions, rows, audits = {}, {}, [], {}
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        print(f"Fitting {candidate_id} on {len(train)} training rows...", flush=True)
        pipeline = fit_training(make_pipeline(candidate, config), X_train, train, config["threads"])
        train_probs = predict_hda(pipeline, X_train, config["threads"])
        validation_probs = predict_hda(pipeline, X_validation, config["threads"])
        rows.append({"candidate_id": candidate_id, "family": candidate["family"], "parameters": json.dumps(candidate["params"], sort_keys=True),
                     **score_predictions(train.FTR.to_numpy(), validation.FTR.to_numpy(), train_probs, validation_probs)})
        imputer = pipeline.named_steps["imputer"]
        expected_medians = X_train.median().fillna(0).to_numpy()
        np.testing.assert_allclose(imputer.statistics_, expected_medians)
        audits[candidate_id] = {
            "training_rows": len(train), "raw_feature_count": len(FEATURE_COLUMNS),
            "model_input_count": int(pipeline.named_steps["model"].n_features_in_),
            "imputer_medians": imputer.statistics_.tolist(),
            "indicator_features": [FEATURE_COLUMNS[i] for i in imputer.indicator_.features_],
            "scaler_mean": pipeline.named_steps["scaler"].mean_.tolist() if "scaler" in pipeline.named_steps else None,
            "estimator_parameters": {key: "NaN" if isinstance(value, float) and np.isnan(value) else value
                                     for key, value in pipeline.named_steps["model"].get_params(deep=False).items()},
        }
        fitted[candidate_id], predictions[candidate_id] = pipeline, validation_probs
    results = pd.DataFrame(rows)
    decision = select_models(results, config["selection"]["tolerance"], tuple(config["selection"]["family_preference"]))
    results["selected_in_family"] = results.candidate_id.isin(decision["family_candidates"].values())
    results["winner"] = results.candidate_id.eq(decision["winner_id"])
    frequency = training_frequency(train)
    baseline = {"candidate_id": "training_frequency", "family": "baseline", "parameters": "training H/D/A frequencies",
                **score_predictions(train.FTR.to_numpy(), validation.FTR.to_numpy(), np.tile(frequency, (len(train), 1)), np.tile(frequency, (len(validation), 1)))}
    baseline["probabilities_h_d_a"] = frequency.tolist()
    results.to_csv(reports / "validation_results.csv", index=False)
    (reports / "frequency_validation.json").write_text(json.dumps(baseline, indent=2) + "\n")
    validation_exports = []
    for candidate_id, probabilities in predictions.items():
        frame = validation[KEY + ["FTR"]].copy()
        frame["candidate_id"], frame["split"] = candidate_id, "validation"
        frame[["p_H", "p_D", "p_A"]] = probabilities
        validation_exports.append(frame)
    pd.concat(validation_exports, ignore_index=True).to_csv(reports / "validation_predictions.csv", index=False)
    selected = {family: fitted[candidate_id] for family, candidate_id in decision["family_candidates"].items()}
    artifact = reports / "selected_pipelines.joblib"
    joblib.dump({"pipelines": selected, "family_candidates": decision["family_candidates"], "winner_id": decision["winner_id"],
                 "class_order": CLASS_ORDER, "feature_columns": FEATURE_COLUMNS, "training_frequency": frequency}, artifact)
    for family, saved_pipeline in joblib.load(artifact)["pipelines"].items():
        np.testing.assert_allclose(predict_hda(saved_pipeline, X_validation, config["threads"]), predictions[decision["family_candidates"][family]])
    decision.update({"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                     "training_rows": len(train), "validation_rows": len(validation)})
    (reports / "winner_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    winner_probs = predictions[decision["winner_id"]]
    example = validation.iloc[0]
    example_json = {
        "candidate_id": decision["winner_id"], "split": "validation",
        "fixture": {**example[KEY].to_dict(), "Date": example.Date.date().isoformat()},
        "actual_result": example.FTR, "probabilities_h_d_a": dict(zip(CLASS_ORDER, winner_probs[0].tolist())),
        "predicted_result": CLASS_ORDER[int(winner_probs[0].argmax())],
        "raw_features": {name: None if pd.isna(value) else float(value) for name, value in X_validation.iloc[0].items()},
    }
    (reports / "validation_example.json").write_text(json.dumps(example_json, indent=2) + "\n")
    metadata.update({"split_dates": {name: {"first": frame.Date.min().date().isoformat(), "last": frame.Date.max().date().isoformat(), "rows": len(frame)} for name, frame in [("train", train), ("validation", validation)]},
                     "preprocessing_audits": audits, "completed_at_utc": decision["frozen_at_utc"]})
    (reports / "experiment_config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(results.to_string(index=False))
    print("Frequency baseline:", baseline)
    print("Winner decision:", decision)
    print("Actual validation prediction:", example_json)
    return results, baseline, decision


if __name__ == "__main__":
    run_experiment()
