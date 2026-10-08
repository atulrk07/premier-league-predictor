"""PRD v2.0 rolling features using only earlier EPL dates.

Run `python -m src.features` for training features and a real-match trace.
Missing history remains missing; no imputation, scaling, or models are fitted.
"""

from collections import defaultdict, deque
import json

import numpy as np
import pandas as pd

from src.data import KEY, REQUIRED, ROOT, SEASONS, load_data

HISTORY_DAYS = 365
REST_CAP_DAYS = 60
TEAM_FEATURES = (
    "points_mean_5", "points_mean_10", "goals_for_mean_5", "goals_against_mean_5",
    "history_count_5", "history_count_10", "rest_days",
)
FEATURE_COLUMNS = tuple(f"{side}_{name}" for side in ("home", "away") for name in TEAM_FEATURES)
TRACE_KEY = ("2014/15", pd.Timestamp("2014-10-18"), "Arsenal", "Hull")
HISTORY_COLUMNS = ["Season", "Date", "HomeTeam", "AwayTeam", "Venue", "Opponent",
                   "GoalsFor", "GoalsAgainst", "Points"]


def _prepare(matches):
    missing = set(KEY + REQUIRED) - set(matches.columns)
    if missing:
        raise ValueError(f"Feature input missing columns: {sorted(missing)}")
    # Explicit input projection: odds, shots, cards, etc. are never read.
    df = matches[list(dict.fromkeys(KEY + REQUIRED))].copy()
    if not df.index.is_unique:
        raise ValueError("Feature input requires a unique row index for fixture alignment")
    if not pd.api.types.is_datetime64_any_dtype(df.Date) or df.Date.isna().any():
        raise ValueError("Feature dates must be parsed nonmissing datetimes from the loader")
    if df.Date.dt.tz is not None or not df.Date.eq(df.Date.dt.normalize()).all():
        raise ValueError("Use timezone-free match dates at midnight, not kickoff timestamps")
    if not df.Season.isin(SEASONS).all() or df[KEY].isna().any().any():
        raise ValueError("Invalid fixture season or key")
    if df.duplicated(KEY).any():
        raise ValueError("Duplicate fixture keys must be resolved by the loader")
    for column in ["HomeTeam", "AwayTeam"]:
        if df[column].astype(str).str.strip().eq("").any():
            raise ValueError("Missing team name")
    if df.HomeTeam.eq(df.AwayTeam).any():
        raise ValueError("Team playing itself")
    for column in ["FTHG", "FTAG"]:
        goals = pd.to_numeric(df[column], errors="coerce")
        if not (np.isfinite(goals) & (goals >= 0) & (goals % 1 == 0)).all():
            raise ValueError("Scores must be finite nonnegative integers")
        df[column] = goals.astype(int)
    expected = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    if not df.FTR.eq(expected).all():
        raise ValueError("Result must agree with full-time goals")
    return df.sort_values(["Date", "Season", "HomeTeam", "AwayTeam"])


def _team_features(history, date):
    """Means use actual window sizes, including incomplete histories."""
    recent = list(history)[-10:]
    five = recent[-5:]
    return {
        "points_mean_5": float(np.mean([row["Points"] for row in five])) if five else np.nan,
        "points_mean_10": float(np.mean([row["Points"] for row in recent])) if recent else np.nan,
        "goals_for_mean_5": float(np.mean([row["GoalsFor"] for row in five])) if five else np.nan,
        "goals_against_mean_5": float(np.mean([row["GoalsAgainst"] for row in five])) if five else np.nan,
        "history_count_5": len(five), "history_count_10": len(recent),
        "rest_days": min((date - recent[-1]["Date"]).days, REST_CAP_DAYS) if recent else np.nan,
    }


def _history_record(match, side):
    home = side == "home"
    return {
        "Season": match.Season, "Date": match.Date,
        "HomeTeam": match.HomeTeam, "AwayTeam": match.AwayTeam,
        "Venue": side, "Opponent": match.AwayTeam if home else match.HomeTeam,
        "GoalsFor": match.FTHG if home else match.FTAG,
        "GoalsAgainst": match.FTAG if home else match.FTHG,
        "Points": 1 if match.FTR == "D" else 3 if match.FTR == ("H" if home else "A") else 0,
    }


def _build(matches, trace_key=None):
    ordered = _prepare(matches)
    histories = defaultdict(deque)
    rows, indices, trace = [], [], None
    for date, daily in ordered.groupby("Date", sort=True):
        cutoff = date - pd.Timedelta(days=HISTORY_DAYS)
        # First pass: all fixtures see the histories available before this date.
        for index, match in daily.iterrows():
            features, snapshot = {}, {}
            for side, team in [("home", match.HomeTeam), ("away", match.AwayTeam)]:
                history = histories[team]
                while history and history[0]["Date"] < cutoff:
                    history.popleft()
                features.update({f"{side}_{name}": value for name, value in _team_features(history, date).items()})
                if trace_key is not None and tuple(match[column] for column in KEY) == trace_key:
                    recent = list(history)[-10:]
                    snapshot[side] = pd.DataFrame(recent, columns=HISTORY_COLUMNS).assign(
                        in_window_5=[i >= len(recent) - 5 for i in range(len(recent))]
                    )
            if snapshot:
                trace = {"fixture": dict(zip(KEY, trace_key)), "features": features, "history": snapshot}
            rows.append(features)
            indices.append(index)
        # Second pass: today's scores become eligible only on a later date.
        for match in daily.itertuples(index=False):
            histories[match.HomeTeam].append(_history_record(match, "home"))
            histories[match.AwayTeam].append(_history_record(match, "away"))
    frame = pd.DataFrame(rows, index=indices, columns=FEATURE_COLUMNS).reindex(matches.index)
    for side in ("home", "away"):
        for window in (5, 10):
            frame[f"{side}_history_count_{window}"] = frame[f"{side}_history_count_{window}"].astype(int)
    return frame, trace


def build_features(matches):
    """Return exactly 14 numeric inputs, aligned to the supplied fixture index.

    Eligible history is D-365 <= history date < D. Home and away appearances
    both count. Cold-start counts are 0; means and rest days are NaN.
    No season reset, label, odds, fixture metadata, or constant home advantage
    column is included in the returned model-input frame.
    """
    return _build(matches)[0]


def trace_fixture(matches, season, date, home_team, away_team):
    """Show the same builder's contributing last-ten histories for one fixture."""
    key = (season, pd.Timestamp(date), home_team, away_team)
    # Later results cannot contribute; don't inspect them for a fixture trace.
    earlier_and_current = matches.loc[matches.Date <= key[1]]
    _, trace = _build(earlier_and_current, trace_key=key)
    if trace is None:
        raise ValueError("Requested fixture not found")
    return trace


def save_feature_report(matches, features, trace):
    """Save a training-only trace, feature matrix, definitions, and source hashes."""
    if not matches.Season.isin(SEASONS[:9]).all():
        raise ValueError("This milestone report accepts training seasons only")
    if tuple(features.columns) != FEATURE_COLUMNS or not features.index.equals(matches.index):
        raise ValueError("Features must use the allowlist and align with fixtures")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    pd.concat([matches[KEY], features], axis=1).to_csv(reports / "features_train.csv", index=False)
    trace_json = {
        "fixture": {**trace["fixture"], "Date": trace["fixture"]["Date"].date().isoformat()},
        "features": {name: None if pd.isna(value) else value for name, value in trace["features"].items()},
        "history": {},
    }
    for side, history in trace["history"].items():
        trace_json["history"][side] = history.assign(Date=history.Date.dt.strftime("%Y-%m-%d")).to_dict("records") if len(history) else []
    (reports / "feature_trace.json").write_text(json.dumps(trace_json, indent=2, allow_nan=False) + "\n")
    sources = json.loads((ROOT / "data/sources.json").read_text())
    summary = {
        "scope": "Training seasons only; no classifier training or held-out analysis.",
        "rows": len(features), "raw_feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS), "history_days": HISTORY_DAYS,
        "eligible_history": "D-365 <= previous match date < D", "rest_cap_days": REST_CAP_DAYS,
        "missing_counts": features.isna().sum().to_dict(),
        "source_sha256": {season: sources[season]["sha256"] for season in sorted(matches.Season.unique())},
        "preprocessing": "None fitted; missing indicators and imputation belong in later training pipelines.",
    }
    (reports / "feature_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    train, _ = load_data(seasons=SEASONS[:9])
    features = build_features(train)
    trace = trace_fixture(train, *TRACE_KEY)
    save_feature_report(train, features, trace)
    print(f"Training feature matrix: {len(features)} rows x {len(features.columns)} raw numeric features")
    print("Fixture:", trace["fixture"])
    for side, history in trace["history"].items():
        print(f"\n{side} history (chronological; in_window_5 marks the shorter window):")
        print(history.to_string(index=False))
    print("\nPre-match features:", trace["features"])


if __name__ == "__main__":
    main()
