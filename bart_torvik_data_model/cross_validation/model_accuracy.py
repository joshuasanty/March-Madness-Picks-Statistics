import os
import pandas as pd
from sklearn.metrics import log_loss

for season in range(2010, 2027):
    actual_path = f"../tournament_games_data/games_{season}.csv"
    pred_path = f"cv_outputs_logistic/all_probabilities_results_{season}.csv"

    if not os.path.exists(actual_path) or not os.path.exists(pred_path):
        continue

    actual_df = pd.read_csv(actual_path)
    simulated_df = pd.read_csv(pred_path)

    # Match games regardless of team order
    actual_df["match_key"] = actual_df.apply(
        lambda row: tuple(sorted([row["Team_W"], row["Team_L"]])),
        axis=1
    )
    simulated_df["match_key"] = simulated_df.apply(
        lambda row: tuple(sorted([row["TeamA"], row["TeamB"]])),
        axis=1
    )

    merged = actual_df.merge(
        simulated_df,
        on="match_key",
        how="inner",
        suffixes=("_actual", "_pred")
    )

    if merged.empty:
        print(f"Season {season}: no matching games found")
        continue

    # Accuracy
    merged["correct"] = merged["Team_W"] == merged["Predicted_Winner"]
    accuracy = merged["correct"].mean()

    # For log loss, define class 1 = TeamB wins
    y_true = (merged["Team_W"] == merged["TeamB"]).astype(int)
    y_pred = merged["Prob_Team_B_Wins"]

    print(f"Season {season} Correct Picks: {merged["correct"].sum()}")
    print(f"Season {season} Accuracy: {100 * accuracy:.4f}%")
    print(f"Season {season} Log Loss: {log_loss(y_true, y_pred):.6f}")

    #Show champion prediction and actual champion
    # Last row = actual champion
    # actual_champion = actual_df.iloc[-1]["Team_W"]

    # Find predicted winner for the same match in merged
    # final_match_key = actual_df.iloc[-1]["match_key"]
    # predicted_champion = merged.loc[merged["match_key"] == final_match_key, "Predicted_Winner"].values[0]

    # print(f"Season {season} Actual Champion: {actual_champion}")
    # print(f"Season {season} Predicted Champion: {predicted_champion}")
