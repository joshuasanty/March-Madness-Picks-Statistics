import shutil
from pathlib import Path

import pandas as pd

from model_training import (
    train_logistic_regression_model,
    train_lasso_logistic_regression_model,
    stats,
)

# ------------------------------
# CONFIG
# ------------------------------
TRAINING_CSV = "training_data/training_data.csv"

CV_START_SEASON = 2010
CV_END_SEASON = 2025

FINAL_TEST_SEASON = 2026
TEAMS_CSV_TEMPLATE = "all_probability_data/{season}.csv"

USE_LOGISTIC = True
USE_LASSO = False

RUN_CV = True
RUN_FINAL_2026 = True

OUTPUT_DIR = Path("cv_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

stat_map = {
    "AdjOE": "AdjOE",
    "AdjDE": "AdjDE",
    "EFG%": "EFG%",
    "EFGD%": "EFGD%",
    "TOR": "TOR",
    "TORD": "TORD",
    "ORB": "ORB",
    "DRB": "DRB",
    "ADJ T": "ADJ T",
    "WAB": "WAB",
}


def build_matchup_df(teams_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, team_a in teams_df.iterrows():
        for _, team_b in teams_df.iterrows():
            if team_a["Team"] >= team_b["Team"]:
                continue  # avoid duplicates and self-matchups

            row = {
                "Season": int(team_a["Season"]),
                "TeamA": team_a["Team"],
                "TeamB": team_b["Team"],
            }

            for stat in stat_map:
                a_val = team_a[stat]
                b_val = team_b[stat]
                row[f"{stat}_diff"] = a_val - b_val

            rows.append(row)

    matchup_df = pd.DataFrame(rows)
    return matchup_df


def train_model_on_csv(csv_path: str):
    if USE_LOGISTIC:
        model, feature_cols = train_logistic_regression_model(
            csv_path=csv_path,
            season_end=9999,   #all seasons
            verbose=False
        )
    else:
        # Keep this if your lasso function returns (model, feature_cols).
        # If it only returns model in your code, keep your current unpacking style.
        model, feature_cols = train_lasso_logistic_regression_model(
            csv_path=csv_path,
            season_end=9999,
            C=1.0,
            verbose=False
        )

    return model, feature_cols


def predict_matchups(model, feature_cols, teams_csv: str) -> pd.DataFrame:
    teams_df = pd.read_csv(teams_csv)
    matchup_df = build_matchup_df(teams_df)

    X = matchup_df[feature_cols]

    matchup_df["Prob_Team_A_Wins"] = model.predict_proba(X)[:, 1]
    matchup_df["Prob_Team_B_Wins"] = 1 - matchup_df["Prob_Team_A_Wins"]
    matchup_df["Predicted_Winner"] = matchup_df.apply(
        lambda row: row["TeamA"] if row["Prob_Team_A_Wins"] > 0.5 else row["TeamB"],
        axis=1
    )

    front_cols = [
        "Season",
        "TeamA",
        "TeamB",
        "Predicted_Winner",
        "Prob_Team_A_Wins",
        "Prob_Team_B_Wins",
    ]
    remaining_cols = [col for col in matchup_df.columns if col not in front_cols]
    matchup_df = matchup_df[front_cols + remaining_cols]

    return matchup_df


def run_one_fold(test_season: int):
    print(f"\n=== Fold: hold out season {test_season} ===")

    full_df = pd.read_csv(TRAINING_CSV)

    # Only use seasons you want in CV/final training
    train_df = full_df[(full_df["Season"] >= CV_START_SEASON) & (full_df["Season"] <= CV_END_SEASON)].copy()
    train_df = train_df[train_df["Season"] != test_season].copy()

    # temp training csv
    temp_train_csv = OUTPUT_DIR / f"_temp_train_excluding_{test_season}.csv"
    train_df.to_csv(temp_train_csv, index=False)

    model, feature_cols = train_model_on_csv(str(temp_train_csv))

    # Keep the training-data CSV that your trainer writes, but rename it per fold
    if Path("actual_training_data.csv").exists():
        shutil.move(
            "actual_training_data.csv",
            OUTPUT_DIR / f"actual_training_data_excluding_{test_season}.csv"
        )

    teams_csv = TEAMS_CSV_TEMPLATE.format(season=test_season)
    predictions_df = predict_matchups(model, feature_cols, teams_csv)

    out_csv = OUTPUT_DIR / f"all_probabilities_results_{test_season}.csv"
    predictions_df.to_csv(out_csv, index=False)

    #cleaning
    temp_train_csv.unlink(missing_ok=True)

    print(f"Saved: {out_csv}")


def run_final_2026():
    print(f"\n=== Final prediction for season {FINAL_TEST_SEASON} ===")

    full_df = pd.read_csv(TRAINING_CSV)
    train_df = full_df[(full_df["Season"] >= CV_START_SEASON) & (full_df["Season"] <= CV_END_SEASON)].copy()

    temp_train_csv = OUTPUT_DIR / f"_temp_train_final_{FINAL_TEST_SEASON}.csv"
    train_df.to_csv(temp_train_csv, index=False)

    model, feature_cols = train_model_on_csv(str(temp_train_csv))

    if Path("actual_training_data.csv").exists():
        shutil.move(
            "actual_training_data.csv",
            OUTPUT_DIR / f"actual_training_data_final_{FINAL_TEST_SEASON}.csv"
        )

    teams_csv = TEAMS_CSV_TEMPLATE.format(season=FINAL_TEST_SEASON)
    predictions_df = predict_matchups(model, feature_cols, teams_csv)

    out_csv = OUTPUT_DIR / f"all_probabilities_results_{FINAL_TEST_SEASON}.csv"
    predictions_df.to_csv(out_csv, index=False)

    temp_train_csv.unlink(missing_ok=True)

    print(f"Saved: {out_csv}")


if RUN_CV:
    for season in range(CV_START_SEASON, CV_END_SEASON + 1):
        run_one_fold(season)

if RUN_FINAL_2026:
    run_final_2026()