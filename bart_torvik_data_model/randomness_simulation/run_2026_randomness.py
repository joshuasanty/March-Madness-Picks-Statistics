#Possibly obselete?
#Another method of picking the most common champion (differs from deterministic-random hybrid

import pandas as pd
import numpy as np

from collections import Counter
from bart_torvik_data_model.model_training import (
    train_logistic_regression_model,
    train_lasso_logistic_regression_model,
    stats,
)

# ------------------------------
# CONFIG
# ------------------------------
TRAINING_CSV = "../training_data/training_data.csv"
BRACKET_FILE = "../tournament_simulation/ordered_games_2026.csv"
USE_LOGISTIC = True
USE_LASSO = False
N_SIMULATIONS = 100
RANDOM_SEED = 42

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
    )
elif USE_LASSO:
    model, feature_cols = train_lasso_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end,
    )
else:
    raise RuntimeError("Set USE_LOGISTIC or USE_LASSO to True.")

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
        diff_dict[f"{s}_diff"] = float(row[f"TeamA_{s}"]) - float(row[f"TeamB_{s}"])
    X = pd.DataFrame([diff_dict])
    return X[feature_cols]

def winner_record_from_row(row, winner_side):
    side = "TeamA" if winner_side == "A" else "TeamB"
    rec = {"Team": row[side]}
    for s in stats:
        rec[s] = row[f"{side}_{s}"]
    return rec

def build_next_round_games(winners):
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

def simulate_one_tournament(starting_bracket, rng):
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

            # random draw using the model probability
            winner_side = "A" if rng.random() < prob_A else "B"
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
            champion = winners[0]["Team"]
            break

        if len(winners) % 2 != 0:
            raise ValueError(
                f"Round {round_num} produced an odd number of winners. "
                "Your bracket is not a valid power-of-two bracket."
            )

        current_round_games = build_next_round_games(winners)
        round_num += 1

    return pd.DataFrame(all_predictions), champion

# ------------------------------
# RUN 100 SIMULATIONS
# ------------------------------
rng = np.random.default_rng(RANDOM_SEED)

champion_counts = Counter()
champion_by_sim = []
all_sim_results = []

for sim in range(N_SIMULATIONS):
    sim_preds, champ = simulate_one_tournament(bracket_df, rng)
    champion_counts[champ] += 1
    champion_by_sim.append({"Simulation": sim + 1, "Champion": champ})
    sim_preds["Simulation"] = sim + 1
    all_sim_results.append(sim_preds)

champions_df = pd.DataFrame(champion_by_sim)
all_predictions_df = pd.concat(all_sim_results, ignore_index=True)

# ------------------------------
# OUTPUT SUMMARY
# ------------------------------
print(f"\nRan {N_SIMULATIONS} simulations for season {test_season}.\n")
print("Most common champions:")
for team, count in champion_counts.most_common(10):
    print(f"{team}: {count} ({count / N_SIMULATIONS:.1%})")

# Optional: save results
all_predictions_df.to_csv("all_simulation_predictions.csv", index=False)
champions_df.to_csv("simulation_champions.csv", index=False)