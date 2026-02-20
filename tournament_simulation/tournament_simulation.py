import pandas as pd
from model_training import train_logistic_regression_model, stats
from collections import Counter
from sklearn.metrics import brier_score_loss
import numpy as np

model, feature_cols, team_stats = train_logistic_regression_model()


def build_feature_vector(stats_A, stats_B):
    row = {f"{s}_diff": stats_A[s] - stats_B[s] for s in stats}
    return pd.DataFrame([row])


def predict_winner(teamA, teamB, model, team_stats):
    stats_A = team_stats.loc[teamA]
    stats_B = team_stats.loc[teamB]

    X = build_feature_vector(stats_A, stats_B)
    prob_A = model.predict_proba(X)[0, 1]

    return teamA if prob_A >= 0.5 else teamB


# ------------------------------
# Step 1: Build starting bracket
# ------------------------------
def build_starting_bracket(tourney_csv, season):
    """
    Returns a Round-of-64 starting bracket as a DataFrame.
    Assumes:
      - Teams are pre-ordered exactly for simulation
      - Columns: Team_W, Team_L
      - First Four games are excluded from the CSV (only R64+)
    """
    r64 = pd.read_csv(tourney_csv)

    bracket = pd.DataFrame({
        "Season": season,
        "GameID": range(1, 33),
        "TeamA": r64["Team_W"],
        "TeamB": r64["Team_L"],
        "Round": 1
    })

    return bracket


def simulate_round(games, model, team_stats):
    winners = []
    for teamA, teamB in games:
        winner = predict_winner(teamA, teamB, model, team_stats)
        winners.append(winner)
    return winners


def pair_next_round(winners):
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def simulate_tournament(bracket, model, team_stats):
    games = list(zip(bracket["TeamA"], bracket["TeamB"]))
    round_num = 1

    all_predictions = []

    while True:

        winners = []

        for teamA, teamB in games:
            winner = predict_winner(teamA, teamB, model, team_stats)
            winners.append(winner)

            all_predictions.append({
                "Season": bracket["Season"].iloc[0],
                "Round": round_num,
                "TeamA": teamA,
                "TeamB": teamB,
                "PredictedWinner": winner
            })

        if len(winners) == 1:
            break
        # Prepare next round by pairing sequentially
        games = pair_next_round(winners)
        round_num += 1

    return pd.DataFrame(all_predictions)


def matchup_key(a, b):
    return tuple(sorted([a, b]))


def predict_probability(teamA, teamB, model, team_stats):
    statsA = team_stats.loc[teamA]
    statsB = team_stats.loc[teamB]

    X = build_feature_vector(statsA, statsB)
    return model.predict_proba(X)[0, 1]


# ----------------------
# SIMULATE TOURNAMENT
# ----------------------
season = 2024
bracket = build_starting_bracket("ordered_games_2024.csv", season)

predictions = simulate_tournament(bracket, model, team_stats)

predictions["Matchup"] = predictions.apply(
    lambda x: matchup_key(x["TeamA"], x["TeamB"]), axis=1
)

# print("Bracket Results:", predictions.to_string())


# ------------------------------------------
# ---------- Evaulating Accuracy ----------
# ------------------------------------------

training_data = pd.read_csv(
    "C:/Users/joshu/PycharmProjects/PythonProject/March-Madness-Picks-Statistics/training_data/training_data.csv")

actual_2024 = training_data[training_data["Season"] == season].copy()
# delete the first 4 games, since I am not simulating the "First Four"
actual_2024 = actual_2024.iloc[4:]

# Count actual wins per team
actual_wins = Counter(actual_2024["Team_W"])

# Count predicted wins per team
predicted_winners = predictions["PredictedWinner"].tolist()
predicted_counts = Counter(predicted_winners)

correct_wins_per_team = {
    team: min(predicted_counts.get(team, 0), actual_wins[team])
    for team in actual_wins
}

num_correct_winners = sum(correct_wins_per_team.values())
total_games = len(actual_2024)
accuracy = num_correct_winners / total_games
print("Total Games:", total_games)
print("Number of Correct Winner Picks:", num_correct_winners)
print("Winner-Pick Accuracy:", accuracy)

# Calculate Log Loss and Brier Score

y_true = []
y_probs = []

for _, row in actual_2024.iterrows():
    winner = row["Team_W"]
    loser = row["Team_L"]

    p = predict_probability(winner, loser, model, team_stats)
    y_true.append(1)
    y_probs.append(p)

losses = -np.log(y_probs)
tournament_log_loss = np.mean(losses)

brier = brier_score_loss(y_true, y_probs)
print("Tournament Brier Score:", brier)
print("Tournament Log Loss:", tournament_log_loss)
print("Tournament Brier Score:", brier, "(top 1% is .109 - the lower the better)")

# print("Min prob:", np.min(y_probs))
# print("Max prob:", np.max(y_probs))
# print("Mean prob:", np.mean(y_probs))
