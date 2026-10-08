"""Small synthetic fixtures protect the data contract, without examining test outcomes."""
import numpy as np
import pandas as pd
import pytest

from src.data import load_data, validate_matches, valid_odds
from src.metrics import (
    LOG_LOSS_EPSILON, baseline_report, bookmaker_probabilities, multiclass_log_loss,
    ranked_probability_score, reorder_probabilities, training_frequency,
)


def fixture():
    return pd.DataFrame([dict(Date="08/09/2014", HomeTeam="Home", AwayTeam="Away",
                              FTHG=2, FTAG=1, FTR="H")])


def test_day_first_and_identical_duplicate():
    df, removed = validate_matches(pd.concat([fixture(), fixture()]), "2014/15")
    assert df.Date.iloc[0] == pd.Timestamp("2014-09-08")
    assert len(df) == 1 and removed == 1


@pytest.mark.parametrize("column,value", [
    ("Date", "31/02/2015"), ("Date", "08/09/2020"),
    ("FTHG", -1), ("FTHG", 1.5), ("FTAG", np.inf), ("FTAG", None),
    ("FTR", "X"), ("FTR", "D"), ("HomeTeam", None), ("AwayTeam", "Home"),
])
def test_invalid_match_is_rejected(column, value):
    with pytest.raises(ValueError):
        validate_matches(fixture().assign(**{column: value}), "2014/15")


def test_missing_required_column():
    with pytest.raises(ValueError, match="missing columns"):
        validate_matches(fixture().drop(columns="FTR"), "2014/15")


def test_blank_rows_only_are_excludable():
    blank = pd.DataFrame([{column: None for column in fixture().columns}])
    df, removed = validate_matches(pd.concat([fixture(), blank]), "2014/15")
    assert len(df) == 1 and removed == 0
    with pytest.raises(ValueError):
        validate_matches(fixture().assign(Date=None), "2014/15")


def test_conflicting_duplicate():
    conflict = fixture().assign(FTHG=3)
    with pytest.raises(ValueError, match="conflicting fixture duplicates"):
        validate_matches(pd.concat([fixture(), conflict]), "2014/15")


def test_missing_odds_do_not_remove_match():
    df, _ = validate_matches(fixture(), "2014/15")
    assert len(df) == 1
    assert not valid_odds(df).iloc[0]
    assert valid_odds(df.assign(B365H=2, B365D=3, B365A=4)).iloc[0]
    assert not valid_odds(df.assign(B365H=1, B365D=3, B365A=4)).iloc[0]


def test_tampered_snapshot_rejected(tmp_path):
    import json

    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    fixture().to_csv(raw / "E0_1415.csv", index=False)
    (tmp_path / "data/sources.json").write_text(json.dumps({"2014/15": {"sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_data(tmp_path)


def test_training_frequency_and_held_out_guard():
    train = pd.DataFrame({"Season": ["2014/15"] * 4, "FTR": ["H", "H", "D", "A"]})
    np.testing.assert_allclose(training_frequency(train), [0.5, 0.25, 0.25])
    for season in ["2023/24", "2024/25", "2025/26", "unknown"]:
        with pytest.raises(ValueError, match="only 2014/15"):
            training_frequency(pd.concat([train, pd.DataFrame({"Season": [season], "FTR": ["A"]})]))
    np.testing.assert_allclose(training_frequency(train.assign(FTR="H")), [1, 0, 0])


def test_bookmaker_hand_example():
    odds = pd.DataFrame({"B365H": [2], "B365D": [3], "B365A": [4]}, index=[42])
    probs = bookmaker_probabilities(odds)
    assert list(probs.columns) == ["H", "D", "A"]
    assert list(probs.index) == [42]
    np.testing.assert_allclose(probs.to_numpy(), [[6 / 13, 4 / 13, 3 / 13]])
    assert probs.sum(axis=1).iloc[0] == pytest.approx(1)
    # Independent scalar hand calculations, not a second copy of the implementation.
    assert multiclass_log_loss(["D"], probs) == pytest.approx(np.log(13 / 4))
    assert ranked_probability_score(["D"], probs) == pytest.approx(45 / 338)


@pytest.mark.parametrize("bad_odds", [None, np.nan, np.inf, 1, 0, -2, "bad"])
def test_invalid_odds_make_entire_triplet_missing(bad_odds):
    odds = pd.DataFrame({"B365H": [2, bad_odds], "B365D": [3, 3], "B365A": [4, 4]}, index=[10, 20])
    result = bookmaker_probabilities(odds)
    assert result.loc[10].notna().all()
    assert result.loc[20].isna().all()
    assert bookmaker_probabilities(odds.drop(columns="B365A")).isna().all().all()


def test_named_probability_order_and_sklearn_agreement():
    from sklearn.metrics import log_loss

    # sklearn uses alphabetical labels A D H; our metrics explicitly use H D A.
    source = np.array([[0.2, 0.3, 0.5], [0.6, 0.25, 0.15]])
    hda = reorder_probabilities(source, ["A", "D", "H"])
    np.testing.assert_allclose(hda, [[0.5, 0.3, 0.2], [0.15, 0.25, 0.6]])
    labels = ["H", "A"]
    assert multiclass_log_loss(labels, hda) == pytest.approx(log_loss(labels, source, labels=["A", "D", "H"]))
    assert ranked_probability_score(labels, hda) == pytest.approx((0.29 / 2 + 0.1825 / 2) / 2)
    with pytest.raises(ValueError, match="exactly once"):
        reorder_probabilities(source, ["H", "H", "A"])


def test_certainty_and_zero_probability_loss():
    for label, row in zip(["H", "D", "A"], np.eye(3)):
        assert multiclass_log_loss([label], [row]) == 0
        assert ranked_probability_score([label], [row]) == 0
    assert ranked_probability_score(["A"], [[1, 0, 0]]) == 1
    assert multiclass_log_loss(["A"], [[1, 0, 0]]) == pytest.approx(-np.log(LOG_LOSS_EPSILON))


@pytest.mark.parametrize("probabilities", [
    [[0.5, 0.3, 0.3]], [[-0.1, 0.5, 0.6]], [[np.nan, 0.5, 0.5]],
    [[np.inf, 0, 0]], [[0.5, 0.5]], [],
])
def test_metrics_reject_invalid_probabilities(probabilities):
    for score in [multiclass_log_loss, ranked_probability_score]:
        with pytest.raises(ValueError):
            score(["H"], probabilities)


def test_metrics_reject_unknown_labels_and_row_mismatch():
    for score in [multiclass_log_loss, ranked_probability_score]:
        for labels in [["X"], [pd.NA], ["H", "D"], []]:
            with pytest.raises(ValueError):
                score(labels, [[0.5, 0.3, 0.2]])


def test_missing_odds_comparison_uses_identical_rows():
    train = pd.DataFrame({"Season": ["2014/15"] * 3, "FTR": ["H", "D", "A"],
                          "B365H": [2, None, 2], "B365D": [3, 3, 3], "B365A": [4, 4, 4]})
    report = baseline_report(train)
    assert report["training_matches"] == 3 and report["bookmaker_matches"] == 2
    all_rows, matched, bookmaker = report["scores"]
    assert all_rows["matches"] == 3
    assert matched["matches"] == bookmaker["matches"] == 2
    assert bookmaker["log_loss"] == pytest.approx((np.log(13 / 6) + np.log(13 / 3)) / 2)
    assert bookmaker["rps"] == pytest.approx((58 / 338 + 136 / 338) / 2)


def test_selected_season_loading_needs_no_held_out_files(tmp_path):
    import hashlib
    import json

    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    path = raw / "E0_1415.csv"
    fixture().to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "data/sources.json").write_text(json.dumps({"2014/15": {"sha256": digest}}))
    matches, summaries = load_data(tmp_path, seasons=["2014/15"])
    assert matches.Season.tolist() == ["2014/15"]
    assert len(summaries) == 1
