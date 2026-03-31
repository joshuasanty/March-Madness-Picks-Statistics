# simulate_bracket_no_results.py
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
TRAINING_CSV = "training_data/training_data.csv"
BRACKET_FILE = "tournament_simulation/ordered_games_2026.csv"
USE_LOGISTIC = True
USE_LASSO = False
SAVE_PREDICTIONS_TO = "predictions_2026.csv"  # set to None to skip saving


# ------------------------------
# LOAD BRACKET
# ------------------------------
bracket_df = pd.read_csv(BRACKET_FILE)

if "Season" in bracket_df.columns and len(bracket_df) > 0:
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
        season_end=train_season_end,
        verbose=True,
    )
elif USE_LASSO:
    model, feature_cols = train_lasso_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end,
        C=1.0,
        verbose=True,
    )
else:
    raise RuntimeError("No model selected. Set USE_LOGISTIC or USE_LASSO to True.")

if not hasattr(model, "predict_proba"):
    raise RuntimeError("Model must support predict_proba().")


# ------------------------------
# VALIDATE INPUT COLUMNS
# ------------------------------
required_cols = ["TeamA", "TeamB"]
for s in stats:
    required_cols.append(f"TeamA_{s}")
    required_cols.append(f"TeamB_{s}")

missing_cols = [c for c in required_cols if c not in bracket_df.columns]
if missing_cols:
    raise ValueError(
        "Bracket CSV missing required columns:\n  " + "\n  ".join(missing_cols)
    )


# ------------------------------
# HELPERS
# ------------------------------
def build_feature_vector_from_row(row):
    diff_dict = {}
    for s in stats:
        a_col = f"TeamA_{s}"
        b_col = f"TeamB_{s}"
        diff_dict[f"{s}_diff"] = float(row[a_col]) - float(row[b_col])

    X = pd.DataFrame([diff_dict])
    X = X[feature_cols]
    print("Test feature df: ", X)
    return X


def winner_record_from_row(row, winner_side):
    """
    winner_side must be 'A' or 'B'.
    Returns a dict with Team plus the winner's stats copied forward.
    """
    side = "TeamA" if winner_side == "A" else "TeamB"
    record = {"Team": row[side]}

    for s in stats:
        record[s] = row[f"{side}_{s}"]

    return record


def build_next_round_games(winners):
    """
    winners is a list of dicts like:
      {"Team": ..., "AdjOE": ..., "AdjDE": ..., ...}
    Returns a DataFrame for the next round with TeamA_/TeamB_ stat columns.
    """
    next_rows = []

    for i in range(0, len(winners), 2):
        A = winners[i]
        B = winners[i + 1]

        new_row = {
            "TeamA": A["Team"],
            "TeamB": B["Team"],
        }

        for s in stats:
            new_row[f"TeamA_{s}"] = A[s]
            new_row[f"TeamB_{s}"] = B[s]

        next_rows.append(new_row)

    return pd.DataFrame(next_rows)


def simulate_tournament(starting_bracket):
    """
    Simulates all rounds until one champion remains.
    starting_bracket must already contain the first round matchups and stats.
    """
    current_round_games = starting_bracket.copy().reset_index(drop=True)
    round_num = 1
    all_predictions = []

    while True:
        winners = []

        for _, row in current_round_games.iterrows():
            teamA = row["TeamA"]
            teamB = row["TeamB"]

            X = build_feature_vector_from_row(row)
            prob_A = float(model.predict_proba(X)[0, 1])
            prob_A = float(np.clip(prob_A, 1e-15, 1 - 1e-15))

            winner_side = "A" if prob_A >= 0.5 else "B"
            winner = teamA if winner_side == "A" else teamB

            all_predictions.append({
                "Season": test_season,
                "Round": round_num,
                "TeamA": teamA,
                "TeamB": teamB,
                "Prob_TeamA_Win": prob_A,
                "PredictedWinner": winner,
            })

            winners.append(winner_record_from_row(row, winner_side))

        if len(winners) == 1:
            break

        if len(winners) % 2 != 0:
            raise ValueError(
                f"Round {round_num} produced an odd number of winners ({len(winners)}). "
                "Your starting bracket is likely not a valid power-of-two tournament bracket."
            )

        current_round_games = build_next_round_games(winners)
        round_num += 1

    return pd.DataFrame(all_predictions)


# ------------------------------
# RUN SIMULATION
# ------------------------------
predictions_df = simulate_tournament(bracket_df)

predictions_df["Matchup"] = predictions_df.apply(
    lambda x: tuple(sorted([x["TeamA"], x["TeamB"]])),
    axis=1
)

if SAVE_PREDICTIONS_TO:
    predictions_df.to_csv(SAVE_PREDICTIONS_TO, index=False)

print(f"Predictions generated for season {test_season}.")
print(predictions_df.to_string(index=False))

print(predictions_df["PredictedWinner"].to_string())