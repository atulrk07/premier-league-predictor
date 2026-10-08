"""Download immutable raw snapshots and validate the PRD's match contract.

Run from the repository root: python -m src.data [--download]
No features or models are created here.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import ssl
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import certifi

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]
KEY = ["Season", "Date", "HomeTeam", "AwayTeam"]
ODDS = ["B365H", "B365D", "B365A"]
SEASONS = [f"{year}/{str(year + 1)[-2:]}" for year in range(2014, 2026)]


def validate_matches(frame: pd.DataFrame, season: str):
    """Reject bad fixtures; remove only fully identical duplicate rows."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"{season}: missing columns {sorted(missing)}")
    df = frame.dropna(how="all").copy()
    df["Season"] = season
    # Explicit formats handle both historical two-digit and newer four-digit years.
    dates = df["Date"].astype("string")
    parsed = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    df["Date"] = parsed.fillna(pd.to_datetime(dates, format="%d/%m/%y", errors="coerce"))
    start = int(season[:4])
    if df["Date"].isna().any() or not df["Date"].between(
        pd.Timestamp(start, 7, 1), pd.Timestamp(start + 1, 8, 31)
    ).all():
        raise ValueError(f"{season}: invalid or out-of-season dates")
    for team in ["HomeTeam", "AwayTeam"]:
        if df[team].isna().any() or df[team].astype(str).str.strip().eq("").any():
            raise ValueError(f"{season}: missing team")
    if df["HomeTeam"].eq(df["AwayTeam"]).any():
        raise ValueError(f"{season}: team playing itself")
    if "Div" in df and not df["Div"].eq("E0").all():
        raise ValueError(f"{season}: unexpected division")
    for score in ["FTHG", "FTAG"]:
        values = pd.to_numeric(df[score], errors="coerce")
        if not (np.isfinite(values) & (values >= 0) & (values % 1 == 0)).all():
            raise ValueError(f"{season}: invalid score in {score}")
        df[score] = values.astype(int)
    if not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError(f"{season}: unknown result")
    expected = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    if not df["FTR"].eq(expected).all():
        raise ValueError(f"{season}: score/result mismatch")
    before = len(df)
    df = df.drop_duplicates()
    if df.duplicated(KEY, keep=False).any():
        raise ValueError(f"{season}: conflicting fixture duplicates")
    return df.sort_values(KEY).reset_index(drop=True), before - len(df)


def valid_odds(df):
    odds = df.reindex(columns=ODDS).apply(pd.to_numeric, errors="coerce")
    return (np.isfinite(odds) & (odds > 1)).all(axis=1)


def load_data(root=ROOT, download=False, seasons=None):
    """Cached bytes must match the manifest; downloads never silently overwrite."""
    seasons = SEASONS if seasons is None else list(seasons)
    if not seasons or len(set(seasons)) != len(seasons) or not set(seasons).issubset(SEASONS):
        raise ValueError("Select distinct seasons from the PRD's 2014/15–2025/26 range")
    root = Path(root)
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    frames, summaries = [], []
    for season in seasons:
        year = int(season[:4])
        code = f"{year % 100:02d}{(year + 1) % 100:02d}"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        path = raw / f"E0_{code}.csv"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; run python -m src.data --download")
            context = ssl.create_default_context(cafile=certifi.where())
            with urlopen(url, timeout=60, context=context) as response:
                payload = response.read()
            # Validate before caching HTML errors or malformed CSV downloads.
            validate_matches(pd.read_csv(BytesIO(payload), encoding="utf-8-sig"), season)
            digest = hashlib.sha256(payload).hexdigest()
            if season in manifest and manifest[season]["sha256"] != digest:
                raise ValueError(f"{season}: provider snapshot changed; review before replacing manifest")
            path.write_bytes(payload)
            if season not in manifest:
                manifest[season] = {
                    "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": digest, "file": str(path.relative_to(root)),
                }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        payload = path.read_bytes()
        if season not in manifest or hashlib.sha256(payload).hexdigest() != manifest[season]["sha256"]:
            raise ValueError(f"{season}: raw file missing provenance or hash mismatch")
        original = pd.read_csv(BytesIO(payload), encoding="utf-8-sig")
        clean, excluded = validate_matches(original, season)
        frames.append(clean)
        summaries.append({
            "season": season, "split": "train" if year <= 2022 else "validation" if year == 2023 else "test",
            "raw_rows": len(original), "rows": len(clean), "identical_duplicates_removed": excluded,
            "blank_rows_removed": int(original.isna().all(axis=1).sum()),
            "columns": len(original.columns), "raw_columns": list(original.columns),
            "first_date": clean.Date.min().date().isoformat(), "last_date": clean.Date.max().date().isoformat(),
            "valid_b365_rows": int(valid_odds(clean).sum()),
            "expected_380": len(clean) == 380,
        })
    return pd.concat(frames, ignore_index=True), summaries


def data_report(matches, summaries):
    train = matches[matches.Season.isin(SEASONS[:9])]
    counts = train.FTR.value_counts().reindex(["H", "D", "A"], fill_value=0)
    example = train.sort_values(KEY).iloc[0][["Season"] + REQUIRED].to_dict()
    example["Date"] = example["Date"].date().isoformat()
    return {
        "total_rows": len(matches), "required_columns": REQUIRED,
        "seasons": summaries, "loaded_columns": list(matches.columns),
        "training_result_counts": counts.to_dict(),
        "training_result_frequencies": (counts / len(train)).to_dict(),
        "training_example": example,
        "test_access": "Structural validation only; no test outcome distributions or examples.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    matches, summaries = load_data(download=args.download)
    report = data_report(matches, summaries)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/data_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(summaries).drop(columns="raw_columns").to_string(index=False))
    print(f"\nTotal validated matches: {len(matches)}")
    print("Required columns:", REQUIRED)
    print("Loaded columns:", list(matches.columns))
    print("Training result counts:", report["training_result_counts"])
    print("Training frequencies:", report["training_result_frequencies"])
    print("Training example:", report["training_example"])


if __name__ == "__main__":
    main()
