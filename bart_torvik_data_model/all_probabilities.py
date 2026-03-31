import pandas as pd
import itertools

from model_training import (
    train_logistic_regression_model,
    train_lasso_logistic_regression_model,
    stats,
)

# ------------------------------
# CONFIG
# ------------------------------
TRAINING_CSV = "training_data/training_data.csv"
TEAMS_CSV = "all_probability_data/2026.csv"
USE_LOGISTIC = True
USE_LASSO = False
SAVE_PREDICTIONS_TO = "all_probabilities_results_2026.csv"

bracket_df = pd.read_csv(TEAMS_CSV)

#need to change this if doing cross validation
# -------------------------
if "Season" in bracket_df.columns and len(bracket_df) > 0:
    test_season = int(bracket_df["Season"].iloc[0])
else:
    test_season = 2026

train_season_end = test_season - 1
#-------------------------

#Train Model:
if USE_LOGISTIC:
    model, feature_cols = train_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end,
        verbose=False
    )
else:
    model = train_lasso_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end,
        C=1.0,
        verbose=False
    )
#debugging - get the coefficients of the model
print(model.coef_)
print(feature_cols)

# ------------------------------
# BUILD UNIQUE MATCHUPS
# ------------------------------
teams_df = pd.read_csv(TEAMS_CSV)
#Predict every single possible combination and save to a csv, with team name information included

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

    # Ensure column order matches training features
    X = matchup_df[feature_cols]

# ------------------------------
# PREDICTIONS
# ------------------------------

matchup_df["Prob_Team_A_Wins"] = model.predict_proba(X)[:, 1]
matchup_df["Prob_Team_B_Wins"] = 1 - matchup_df["Prob_Team_A_Wins"]

matchup_df["Predicted_Winner"] = matchup_df.apply(lambda row: row["TeamA"] if row["Prob_Team_A_Wins"] > 0.5 else row["TeamB"], axis=1)


# ------------------------------
# REORDER COLUMNS
# ------------------------------

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
# ------------------------------
# SAVE
# ------------------------------
matchup_df.to_csv(SAVE_PREDICTIONS_TO, index=False)











