"""Historical results dashboard. Launch: python -m streamlit run streamlit_app.py"""

import os

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_data import (
    BOOKMAKER, FEATURE_LABELS, FORECAST_LABELS, PROBABILITIES,
    baseline_improvements, fixture_scores, load_snapshot, outcome_analysis, snapshot_signature,
)
from src.data import ROOT
from src.features import TEAM_FEATURES
from src.metrics import CLASS_ORDER

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/.matplotlib"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTCOMES = {"H": "Home win", "D": "Draw", "A": "Away win"}


@st.cache_data(show_spinner="Loading verified historical reports…", max_entries=2)
def cached_data(signature):
    return load_snapshot()


@st.cache_data(show_spinner=False, max_entries=4)
def cached_analysis(frame):
    return outcome_analysis(frame)


def results_table(frame):
    return frame.rename(columns={"forecast": "Forecast", "matches": "Matches", "log_loss": "Log loss", "rps": "RPS"}).assign(
        Forecast=frame.forecast.map(FORECAST_LABELS), **{"Accuracy (%)": 100 * frame.accuracy},
    )[["Forecast", "Matches", "Log loss", "RPS", "Accuracy (%)"]]


def show_overview(data, season):
    st.subheader("Results overview")
    scores = data["results"].loc[data["results"].season.eq(season)]
    matched = scores.loc[scores.subset.eq("matched_bookmaker")]
    coverage = data["coverage"].loc[data["coverage"].season.eq(season)].iloc[0]
    st.caption(f"{season.title()} · {coverage.bookmaker_matches:,}/{coverage.test_matches:,} fixtures with valid bookmaker odds "
               f"({coverage.coverage:.1%}). The main comparison uses those identical matched fixtures.")
    winner = matched.loc[matched.forecast.eq(data["decision"]["winner_id"])].iloc[0]
    for column, label, value in zip(st.columns(3), ("Log loss ↓", "RPS ↓", "Accuracy ↑"),
                                   (f"{winner.log_loss:.6f}", f"{winner.rps:.6f}", f"{winner.accuracy:.2%}")):
        column.metric(label, value)
    st.dataframe(results_table(matched), hide_index=True, width="stretch")
    st.markdown("**Selected model's improvement over each baseline**")
    st.caption("Calculated from saved scores. Positive means improvement: baseline minus model for losses; "
               "model minus baseline for accuracy. Relative loss reduction divides by baseline loss.")
    st.dataframe(baseline_improvements(matched, data["decision"]["winner_id"]), hide_index=True, width="stretch")
    with st.expander("What do the metrics mean?"):
        st.markdown("**Log loss** penalizes low probability assigned to the actual result. Lower is better; "
                    "it selected the model.\n\n**RPS** measures squared cumulative-probability errors in H/D/A order. "
                    "Lower is better; its assumed outcome ordering is debated.\n\n**Accuracy** measures how often "
                    "the highest-probability outcome is correct. It does not measure probability quality.")
    with st.expander("All model-eligible test matches"):
        st.dataframe(results_table(scores.loc[scores.subset.eq("all_test")]), hide_index=True, width="stretch")
        st.caption("Bookmaker scores are shown in the matched comparison above, so missing odds never change model eligibility.")
    with st.expander("Validation selection: all eight configurations"):
        st.dataframe(data["validation"][["candidate_id", "train_log_loss", "validation_log_loss", "validation_rps",
                                          "validation_accuracy", "selected_in_family", "winner"]], hide_index=True, width="stretch")
        frequency = data["frequency"]
        st.caption(f"Training-frequency validation reference: log loss {frequency['validation_log_loss']:.6f}, "
                   f"RPS {frequency['validation_rps']:.6f}, accuracy {frequency['validation_accuracy']:.2%}.")
        st.write("Minimum validation log loss per family; within 0.005 of the best family candidate, "
                 "prefer logistic regression, then Random Forest, then XGBoost. Only LR C=0.1 qualified.")


def show_explorer(data):
    st.subheader("Historical match explorer")
    matches = data["matches"]
    seasons = sorted(matches.Season.unique())
    season = st.selectbox("Match season", seasons, index=seasons.index("2024/25"), key="match_season")
    fixtures = matches.loc[matches.Season.eq(season)]
    teams = sorted(set(fixtures.HomeTeam) | set(fixtures.AwayTeam))
    team = st.selectbox("Team", ["All teams", *teams], key="team")
    if team != "All teams":
        fixtures = fixtures.loc[fixtures.HomeTeam.eq(team) | fixtures.AwayTeam.eq(team)]
    fixtures = fixtures.sort_values(["Date", "HomeTeam", "AwayTeam"]).set_index("fixture_id", drop=False)
    selected = st.selectbox("Fixture", fixtures.index.tolist(), key="fixture",
                            format_func=lambda key: f"{fixtures.loc[key, 'Date']:%d %b %Y} · "
                            f"{fixtures.loc[key, 'HomeTeam']} vs {fixtures.loc[key, 'AwayTeam']}")
    row = fixtures.loc[selected]
    st.markdown(f"### {row.HomeTeam} vs {row.AwayTeam}")
    st.caption(f"Historical pre-match prediction · {row.Date:%d %B %Y} · "
               f"{'Validation' if row.split == 'validation' else 'Final test'} · LR C=0.1")
    probabilities = pd.DataFrame({"Selected model": row[PROBABILITIES].to_numpy(dtype=float)},
                                 index=[f"{OUTCOMES[outcome]} ({outcome})" for outcome in CLASS_ORDER])
    if row[BOOKMAKER].notna().all():
        probabilities["Normalized Bet365"] = row[BOOKMAKER].to_numpy(dtype=float)
    else:
        st.info("Bookmaker probabilities are unavailable: a complete valid odds triplet is required.")
    for column, outcome, probability in zip(st.columns(3), CLASS_ORDER, row[PROBABILITIES]):
        column.metric(f"{OUTCOMES[outcome]} ({outcome})", f"{probability:.2%}")
    st.bar_chart(probabilities, stack=False)
    st.dataframe((100 * probabilities).rename(columns=lambda name: name + " (%)"), width="stretch")
    predicted = CLASS_ORDER[int(np.argmax(row[PROBABILITIES].to_numpy(dtype=float)))]
    loss, rps = fixture_scores(row)
    st.write(f"**Actual score:** {row.HomeTeam} {row.FTHG}–{row.FTAG} {row.AwayTeam}. "
             f"**Outcome:** {OUTCOMES[row.FTR]} ({row.FTR}). "
             f"**Most likely forecast:** {OUTCOMES[predicted]} ({predicted}).")
    st.caption(f"Individual log loss: {loss:.6f} · RPS: {rps:.6f}. "
               "A correct highest-probability outcome can still have nonzero loss.")
    st.markdown("**Features available before this match**")
    feature_table = pd.DataFrame({
        "Feature": [FEATURE_LABELS[name] for name in TEAM_FEATURES],
        row.HomeTeam: [row[f"home_{name}"] for name in TEAM_FEATURES],
        row.AwayTeam: [row[f"away_{name}"] for name in TEAM_FEATURES],
    })
    st.dataframe(feature_table, hide_index=True, width="stretch")
    st.caption("These are raw inputs. Missing means/rest days use the frozen training medians and missing indicators; "
               "history counts remain visible. Rest is capped at 60 days. Inputs are not feature-contribution scores.")
    st.download_button("Download this historical forecast", fixtures.loc[[selected]].to_csv(index=False).encode(),
                       file_name="historical_forecast.csv", mime="text/csv")


def show_analysis(data, season):
    st.subheader("Model analysis")
    frame = data["matches"].loc[data["matches"].split.eq("test")]
    if season != "combined":
        frame = frame.loc[frame.Season.eq(season)]
    bins, matrix, performance = cached_analysis(frame)
    if season == "combined":
        bins = data["calibration"]  # Reuse the original combined-test bin artifact.
    st.caption(f"{season.title()} · selected logistic regression · {len(frame):,} test fixtures. "
               "Descriptive analysis; no calibration fitting or model reselection.")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]})
    for index, outcome in enumerate(CLASS_ORDER):
        groups = bins.loc[bins.outcome.eq(outcome)]
        occupied = groups.loc[groups["count"] > 0]
        ax = axes[0, index]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(occupied.mean_probability, occupied.observed_frequency, "o-", color="#2563eb")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{OUTCOMES[outcome]} ({outcome})", xlabel="Mean forecast", ylabel="Observed frequency")
        ax.grid(alpha=0.2)
        bars = axes[1, index]
        bars.bar((groups.lower + groups.upper) / 2, groups["count"], width=0.17, color="#93c5fd")
        bars.set(xlim=(0, 1), xlabel="Probability bin", ylabel="Matches")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The dashed diagonal indicates agreement between forecasts and outcomes. "
               "Points above it indicate underprediction; below it, overprediction. Empty bins have no point; small bins are noisy.")
    with st.expander("Calibration bins and counts"):
        st.dataframe(bins, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.dataframe(matrix, width="stretch")
        st.caption("Rows are actual outcomes; columns are highest-probability predictions. H/D/A ordering is explicit.")
    with right:
        st.markdown("**Outcome-specific performance**")
        st.dataframe(performance, hide_index=True, width="stretch")
        st.caption("Support counts actual outcomes; predicted_count counts chosen outcomes. Undefined ratios use zero.")
    st.info("Per-match feature-attribution explanations were not implemented in the saved experiment. "
            "The explorer shows the actual feature inputs without inventing attribution scores.")


def show_methodology(data):
    st.subheader("Methodology")
    st.markdown("Data: [football-data.co.uk England archives](https://www.football-data.co.uk/englandm.php), "
                "division E0, 2014/15–2025/26. Raw bytes are checked against recorded SHA256 hashes.")
    st.dataframe(pd.DataFrame([
        ["Training", "2014/15–2022/23", 3420, "Fit preprocessing and models"],
        ["Validation", "2023/24", 380, "Select configuration and winner"],
        ["Final test", "2024/25–2025/26", 760, "Score frozen selections"],
    ], columns=["Partition", "Seasons", "Matches", "Purpose"]), hide_index=True, width="stretch")
    st.write("Seven features per team: points means over 5/10 matches, goals for/against over 5, "
             "history counts over 5/10, and rest days capped at 60. Averages use actual available counts.")
    st.write("Eligible results satisfy D−365 days ≤ result date < D. All same-day fixtures are computed before "
             "updating history. Earlier test results may inform later fixtures; fitted parameters remain fixed.")
    st.write("The feature allowlist excludes odds and current-match scores, shots, cards, and half-time statistics. "
             "Median imputation and missing indicators are fitted only on training rows; logistic regression also uses "
             "training-fitted scaling. Fourteen raw features become 24 processed inputs.")
    st.write("Bookmaker probabilities normalize inverse B365H/B365D/B365A odds. Invalid odds affect comparison "
             "coverage, not model eligibility. This does not establish true fair probabilities or betting profit.")
    st.markdown("**Limitations:** only one validation and two test seasons; EPL-only history; no injury, lineup, "
                "or player information; limited history for promoted teams; historical provider corrections and odds timing. "
                "Draw recall is weak. Calibration and error analysis are descriptive.")
    st.caption(f"Training cutoff: {data['metadata']['split_dates']['train']['last']}. "
               f"Model artifact SHA256: {data['decision']['artifact_sha256']}. "
               "The original frozen decision and reported results are unchanged.")


def main():
    st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")
    st.title("Premier League Predictor")
    st.caption("Historical results · pre-match probabilities · frozen experiment")
    try:
        data = cached_data(snapshot_signature())
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Historical dashboard data could not be loaded: {error}")
        st.code("python -m src.dashboard_data", language="bash")
        st.write("Prepare the dashboard snapshot from the recorded raw archives. Artifact mismatches require review.")
        st.stop()
    st.info(f"Validation-selected model: logistic regression C=0.1 · "
            f"validation log loss {data['decision']['winner_validation_log_loss']:.6f}. "
            "The winner was frozen before test evaluation.")
    season = st.sidebar.selectbox("Test results season", ["combined", "2024/25", "2025/26"],
                                 format_func=lambda value: "Combined (760 matches)" if value == "combined" else value,
                                 key="results_season")
    st.sidebar.caption("Applies to the results overview and model analysis. The match explorer has its own filters.")
    overview, explorer, analysis, methodology = st.tabs([
        "Results overview", "Historical match explorer", "Model analysis", "Methodology",
    ])
    with overview:
        show_overview(data, season)
    with explorer:
        show_explorer(data)
    with analysis:
        show_analysis(data, season)
    with methodology:
        show_methodology(data)


if __name__ == "__main__":
    main()
