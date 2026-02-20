import pandas as pd
from model_training import train_logistic_regression_model, stats

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
    bracket_results = []

    winner_list = []

    while len(games) >= 1:
        winners = simulate_round(games, model, team_stats)
        bracket_results.append({
            "Round": round_num,
            "Games": games,
            "Winners": winners
        })
        winner_list.append(winners)

        if len(winners) == 1:
            # Champion determined
            champion = winners[0]
            break
        # Prepare next round by pairing sequentially
        games = pair_next_round(winners)
        round_num += 1

    return champion, bracket_results, winner_list


bracket = build_starting_bracket("ordered_games_2024.csv", 2024)
print(bracket) #debugging


champion, bracket_results, winner_list = simulate_tournament(bracket, model, team_stats)
print("Champion:", champion)
print("Bracket Results:", bracket_results)

correct = 0
total = 0

for r in bracket_results:
    for (teamA, teamB), winner in zip(r["Games"], r["Winners"]):
        if winner == bracket_results[-1]["Winners"][0]:
            correct += 1
        total += 1

game_accuracy = correct / total
print("Game Accuracy:", game_accuracy)

print(winner_list)
