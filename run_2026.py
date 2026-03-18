# simulate_bracket_no_results.py
"""
Predict winners for a starting bracket file that contains per-team stats (no results).
Expected BRACKET_FILE layout (one row per game):
TeamA,TeamA_AdjOE,TeamA_AdjDE,TeamA_BARTHAG,TeamA_EFG%,...,TeamB,TeamB_AdjOE,TeamB_AdjDE,TeamB_BARTHAG,TeamB_EFG%,...,Season(optional)

This script:
 - trains a model on seasons <= (test_season - 1)
 - reads the bracket (test season)
 - computes feature vector as TeamA_stat - TeamB_stat
 - predicts probability TeamA wins and the predicted winner
 - saves predictions to CSV if SAVE_PREDICTIONS_TO is set
"""
import pandas as pd
import numpy as np

from model_training import (
    train_logistic_regression_model,
    train_lasso_logistic_regression_model,
    stats,
)

# ------------------------------
# CONFIG
# ------------------------------
TRAINING_CSV = "C:/Users/joshu/PycharmProjects/PythonProject/March-Madness-Picks-Statistics/training_data/training_data.csv"
BRACKET_FILE = "tournament_simulation/ordered_games_2026.csv"     # file with TeamA/TeamB and their stats (no wins/losses)
USE_LOGISTIC = True
USE_LASSO = False
SAVE_PREDICTIONS_TO = "predictions_2026.csv"  # set to None to avoid writing

# ------------------------------
# READ BRACKET & INFER SEASON
# ------------------------------
bracket_df = pd.read_csv(BRACKET_FILE)

# infer test season if present; fallback to 2026
if "Season" in bracket_df.columns:
    test_season = int(bracket_df["Season"].iloc[0])
else:
    test_season = 2026

train_season_end = test_season - 1

# ------------------------------
# TRAIN MODEL
# ------------------------------
if USE_LOGISTIC:
    model, feature_cols = train_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end
    )
elif USE_LASSO:
    model, feature_cols = train_lasso_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end
    )
else:
    raise RuntimeError("No model selected. Set USE_LOGISTIC or USE_LASSO to True.")

# Confirm the model supports probability output
if not hasattr(model, "predict_proba"):
    raise RuntimeError("Trained model does not support predict_proba. Use a probabilistic classifier.")

# ------------------------------
# VALIDATE BRACKET COLUMNS
# ------------------------------
# Required per-stat columns: TeamA_{stat} and TeamB_{stat} for every stat in model_training.stats
required_cols = []
for s in stats:
    required_cols.append(f"TeamA_{s}")
    required_cols.append(f"TeamB_{s}")

missing_cols = [c for c in required_cols if c not in bracket_df.columns]
if missing_cols:
    raise ValueError(
        "Bracket CSV missing required columns for prediction. Missing:\n  "
        + "\n  ".join(missing_cols)
    )

if "TeamA" not in bracket_df.columns or "TeamB" not in bracket_df.columns:
    raise ValueError("Bracket CSV must contain 'TeamA' and 'TeamB' columns with team names.")

# ------------------------------
# BUILD FEATURE VECTOR FROM ROW
# ------------------------------
def feature_vector_from_bracket_row(row):
    """
    Returns a 1-row DataFrame with columns ordered as feature_cols expected by the model.
    Expects bracket CSV to contain TeamA_<stat> and TeamB_<stat> for each stat in stats.
    """
    diff_dict = {}
    for s in stats:
        a_col = f"TeamA_{s}"
        b_col = f"TeamB_{s}"
        # convert to float (raises if non-numeric)
        try:
            a_val = float(row[a_col])
            b_val = float(row[b_col])
        except Exception as e:
            raise ValueError(f"Non-numeric value in row for columns {a_col}/{b_col}: {e}")
        diff_dict[f"{s}_diff"] = a_val - b_val

    # ensure feature_cols are present and ordered as during training
    missing_feats = [c for c in feature_cols if c not in diff_dict]
    if missing_feats:
        raise ValueError(
            "Constructed feature vector missing expected feature columns:\n  "
            + "\n  ".join(missing_feats)
        )

    # create DataFrame and order columns exactly as feature_cols
    X = pd.DataFrame([diff_dict], columns=feature_cols)
    return X

# ------------------------------
# PREDICT
# ------------------------------
pred_rows = []
for i, row in bracket_df.iterrows():
    teamA = row["TeamA"]
    teamB = row["TeamB"]

    X = feature_vector_from_bracket_row(row)
    proba_A = model.predict_proba(X)[0, 1]   # probability TeamA wins
    # clip for numerical stability if needed
    proba_A = float(np.clip(proba_A, 1e-15, 1 - 1e-15))
    predicted = teamA if proba_A >= 0.5 else teamB

    pred_rows.append({
        "Season": test_season,
        "GameIndex": i + 1,
        "TeamA": teamA,
        "TeamB": teamB,
        "Prob_TeamA_Win": proba_A,
        "PredictedWinner": predicted
    })

predictions_df = pd.DataFrame(pred_rows)

# Optional: simulate the bracket forward using same pairing logic as your other script.
# NOTE: this placeholder carries forward winners without recalculating probabilities for later rounds.
def pair_next_round(winners):
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

def simulate_from_starting_predictions(pred_df):
    """
    Simple simulation that uses predicted winners from the starting R64 and advances them by pairing.
    This does NOT recompute probabilities for later rounds. For full probabilistic simulation,
    you need season team-stats (per-team) and to recompute feature vectors for new matchups.
    """
    winners = pred_df["PredictedWinner"].tolist()
    round_num = 1
    all_preds = pred_df.copy()
    all_preds["Round"] = 1

    while len(winners) > 1:
        pairs = pair_next_round(winners)
        next_winners = []
        for a, b in pairs:
            # placeholder: choose first in pair (a). Replace with recomputed logic if team stats are available.
            next_winners.append(a)
        winners = next_winners
        round_num += 1
    return all_preds

# Save or print
if SAVE_PREDICTIONS_TO:
    predictions_df.to_csv(SAVE_PREDICTIONS_TO, index=False)

# display a short summary
print(f"Predictions generated for test season {test_season} — {len(predictions_df)} games.")
print(predictions_df.head(10).to_string(index=False))