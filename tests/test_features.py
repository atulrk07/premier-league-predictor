"""Hand calculations and score mutations protect the rolling-feature contract."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, build_features, trace_fixture


def matches(rows):
    df = pd.DataFrame(rows, columns=["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df.Date)
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    return df


def test_hand_history_home_away_and_cold_start():
    df = matches([
        ("2022/23", "2022-08-01", "A", "B", 2, 0),
        ("2022/23", "2022-08-08", "C", "A", 1, 1),
        ("2022/23", "2022-08-15", "A", "D", 0, 3),
        ("2022/23", "2022-08-22", "A", "B", 0, 0),
    ])
    features = build_features(df)
    assert tuple(features.columns) == FEATURE_COLUMNS and len(FEATURE_COLUMNS) == 14
    cold = features.iloc[0]
    assert cold.home_history_count_5 == cold.away_history_count_10 == 0
    assert cold.drop(labels=[column for column in FEATURE_COLUMNS if "history_count" in column]).isna().all()
    row = features.iloc[3]
    assert row.home_points_mean_5 == row.home_points_mean_10 == pytest.approx(4 / 3)
    assert row.home_goals_for_mean_5 == 1
    assert row.home_goals_against_mean_5 == pytest.approx(4 / 3)
    assert row.home_history_count_5 == row.home_history_count_10 == 3
    assert row.home_rest_days == 7
    assert row.away_points_mean_5 == 0 and row.away_goals_against_mean_5 == 2
    assert row.away_history_count_5 == 1 and row.away_rest_days == 21
    trace = trace_fixture(df, "2022/23", "2022-08-22", "A", "B")
    assert trace["history"]["home"].Points.tolist() == [3, 1, 0]
    assert trace["history"]["home"].Venue.tolist() == ["home", "away", "home"]
    assert all(trace["history"]["home"].Date < pd.Timestamp("2022-08-22"))


def test_five_and_ten_windows_are_distinct():
    start = pd.Timestamp("2022-08-01")
    rows = [("2022/23", start + pd.Timedelta(days=7 * i), "A", f"B{i}",
             0 if i < 6 else 2, 1 if i < 6 else 0) for i in range(11)]
    rows.append(("2022/23", start + pd.Timedelta(days=77), "A", "New", 0, 0))
    df = matches(rows)
    row = build_features(df).iloc[-1]
    assert row.home_points_mean_5 == 3
    assert row.home_points_mean_10 == 1.5
    assert row.home_goals_for_mean_5 == 2 and row.home_goals_against_mean_5 == 0
    assert row.home_history_count_5 == 5 and row.home_history_count_10 == 10
    assert row.home_rest_days == 7
    assert row.away_history_count_10 == 0 and pd.isna(row.away_rest_days)
    history = trace_fixture(df, "2022/23", start + pd.Timedelta(days=77), "A", "New")["history"]["home"]
    assert len(history) == 10 and history.in_window_5.sum() == 5


def test_cross_season_365_day_boundary_expiry_and_rest_cap():
    df = matches([
        ("2022/23", "2022-07-31", "A", "B", 9, 0),  # 366 days ago: expired
        ("2022/23", "2022-08-01", "A", "C", 1, 1),  # exactly 365: eligible
        ("2023/24", "2023-08-01", "A", "B", 0, 0),
    ])
    row = build_features(df).iloc[2]
    assert row.home_history_count_10 == 1 and row.home_points_mean_10 == 1
    assert row.home_goals_for_mean_5 == row.home_goals_against_mean_5 == 1
    assert row.home_rest_days == 60
    assert row.away_history_count_5 == 0 and pd.isna(row.away_rest_days)


def leakage_fixture():
    # Shared-team same-day fixtures stress the batch rule even if input order changes.
    return matches([
        ("2022/23", "2022-08-01", "A", "X", 1, 0),
        ("2022/23", "2022-08-01", "B", "Y", 0, 1),
        ("2022/23", "2022-08-08", "A", "C", 1, 1),
        ("2022/23", "2022-08-08", "A", "B", 2, 0),  # target
        ("2022/23", "2022-08-15", "A", "B", 0, 0),
    ])


@pytest.mark.parametrize("changed_index", [2, 3, 4], ids=["same_day", "current_match", "future_match"])
def test_score_mutations_do_not_change_target_features(changed_index):
    original = leakage_fixture()
    changed = original.copy()
    changed.loc[changed_index, ["FTHG", "FTAG", "FTR"]] = [0, 9, "A"]
    before = build_features(original)
    after = build_features(changed)
    # Every fixture at or before the changed date must remain unchanged.
    eligible = original.Date <= original.loc[changed_index, "Date"]
    pd.testing.assert_frame_equal(before.loc[eligible], after.loc[eligible])
    pd.testing.assert_series_equal(before.loc[3], after.loc[3])
    if changed_index != 4:
        assert before.loc[4, "home_points_mean_5"] != after.loc[4, "home_points_mean_5"]


def test_input_order_and_irrelevant_columns_cannot_change_features():
    original = leakage_fixture()
    changed = original.assign(B365H=999, HS=100, HTHG=88, HY=99)
    pd.testing.assert_frame_equal(build_features(original), build_features(changed))
    shuffled = changed.sample(frac=1, random_state=42)
    pd.testing.assert_frame_equal(build_features(original), build_features(shuffled).reindex(original.index))


def test_earlier_test_results_can_inform_later_test_fixtures():
    df = matches([
        ("2024/25", "2024-08-01", "A", "B", 1, 0),
        ("2024/25", "2024-08-08", "A", "B", 0, 0),
    ])
    row = build_features(df).iloc[1]
    assert row.home_points_mean_5 == 3 and row.away_points_mean_5 == 0
    assert row.home_rest_days == row.away_rest_days == 7


def test_trace_matches_feature_row_and_unknown_fixture_fails():
    df = leakage_fixture()
    trace = trace_fixture(df, "2022/23", "2022-08-08", "A", "B")
    np.testing.assert_allclose(list(trace["features"].values()), build_features(df).loc[3].to_numpy(), equal_nan=True)
    assert trace["history"]["home"].Date.tolist() == [pd.Timestamp("2022-08-01")]
    with pytest.raises(ValueError, match="not found"):
        trace_fixture(df, "2022/23", "2022-08-08", "Missing", "B")
