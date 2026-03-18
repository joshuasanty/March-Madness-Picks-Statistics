import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import brier_score_loss

from model_training import (
    train_logistic_regression_model,
    train_lasso_logistic_regression_model,
    build_team_stats,
    stats
)

# ------------------------------
# CONFIG
# ------------------------------
CSV_PATH = "C:/Users/joshu/PycharmProjects/PythonProject/March-Madness-Picks-Statistics/training_data/training_data.csv"
TOURNEY_FILE = "ordered_games_2024.csv"  # change as needed

USE_LOGISTIC = True
USE_LASSO = False

# ------------------------------
# LOAD DATA + DETERMINE SPLIT
# ------------------------------
df = pd.read_csv(CSV_PATH)

# automatically pick last season as test
test_season = df["Season"].max()
train_season_end = test_season - 1

# ------------------------------
# TRAIN MODEL
# ------------------------------
if USE_LOGISTIC:
    model, feature_cols = train_logistic_regression_model(
        csv_path=CSV_PATH,
        season_end=train_season_end
    )
elif USE_LASSO:
    model, feature_cols = train_lasso_logistic_regression_model(
        csv_path=CSV_PATH,
        season_end=train_season_end
    )

# build team stats ONLY for test season
team_stats = build_team_stats(df, season=test_season)

# ------------------------------
# FEATURE + PREDICTION HELPERS
# ------------------------------
def build_feature_vector(stats_A, stats_B):
    row = {f"{s}_diff": stats_A[s] - stats_B[s] for s in stats}
    return pd.DataFrame([row])[feature_cols]  # enforce column order


def predict_winner(teamA, teamB):
    stats_A = team_stats.loc[teamA]
    stats_B = team_stats.loc[teamB]

    X = build_feature_vector(stats_A, stats_B)
    prob_A = model.predict_proba(X)[0, 1]

    return teamA if prob_A >= 0.5 else teamB


def predict_probability(teamA, teamB):
    stats_A = team_stats.loc[teamA]
    stats_B = team_stats.loc[teamB]

    X = build_feature_vector(stats_A, stats_B)
    return model.predict_proba(X)[0, 1]


# ------------------------------
# BRACKET BUILDING (GENERIC)
# ------------------------------
def build_starting_bracket(tourney_csv, season):
    df = pd.read_csv(tourney_csv)

    # auto-detect column naming
    if "Team_W" in df.columns:
        colA, colB = "Team_W", "Team_L"
    else:
        colA, colB = "Team_A", "Team_B"

    return pd.DataFrame({
        "Season": season,
        "GameID": range(1, len(df) + 1),
        "TeamA": df[colA],
        "TeamB": df[colB],
        "Round": 1
    })


def pair_next_round(winners):
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def simulate_tournament(bracket):
    games = list(zip(bracket["TeamA"], bracket["TeamB"]))
    round_num = 1
    all_predictions = []

    while True:
        winners = []

        for teamA, teamB in games:
            winner = predict_winner(teamA, teamB)
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

        games = pair_next_round(winners)
        round_num += 1

    return pd.DataFrame(all_predictions)


def matchup_key(a, b):
    return tuple(sorted([a, b]))


# ------------------------------
# RUN SIMULATION
# ------------------------------
bracket = build_starting_bracket(TOURNEY_FILE, test_season)
predictions = simulate_tournament(bracket)

predictions["Matchup"] = predictions.apply(
    lambda x: matchup_key(x["TeamA"], x["TeamB"]), axis=1
)

# ------------------------------
# EVALUATION
# ------------------------------
actual = df[df["Season"] == test_season].copy()
print(predictions)
# drop First Four automatically if present
if len(actual) > 63:
    actual = actual.iloc[-63:]

actual_wins = Counter(actual["Team_W"])
predicted_counts = Counter(predictions["PredictedWinner"])

correct_wins_per_team = {
    team: min(predicted_counts.get(team, 0), actual_wins[team])
    for team in actual_wins
}

num_correct = sum(correct_wins_per_team.values())
total_games = len(actual)
accuracy = num_correct / total_games

print("Test Season:", test_season)
print("Total Games:", total_games)
print("Correct Picks:", num_correct)
print("Accuracy:", accuracy)

# ------------------------------
# LOG LOSS + BRIER
# ------------------------------
y_true = []
y_probs = []

for _, row in actual.iterrows():
    p = predict_probability(row["Team_W"], row["Team_L"])
    y_true.append(1)
    y_probs.append(p)

y_probs = np.clip(y_probs, 1e-15, 1 - 1e-15)

log_loss = -np.mean(np.log(y_probs))
brier = brier_score_loss(y_true, y_probs)

print("Log Loss:", log_loss)
print("Brier Score:", brier)