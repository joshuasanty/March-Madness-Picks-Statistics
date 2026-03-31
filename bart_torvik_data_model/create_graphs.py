import matplotlib.pyplot as plt
import pandas as pd

def scatter_plot(x, y, title, xlabel, ylabel, regression=False):
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    if regression:
        plt.plot(x, y, 'r')
        plt.show()

plots = [
    {# Offensive Difference vs Win Probability
        "x": "AdjOE_diff",
        "y": "Prob_Team_A_Wins",
        "title": "Offensive Difference vs Win Probability",
        "xlabel": "Offensive Difference",
        "ylabel": "Win Probability",
    },
    {# Defensive Difference vs Win Probability
        "x": "AdjDE_diff",
        "y": "Prob_Team_A_Wins",
        "title": "Defensive Difference vs Win Probability",
        "xlabel": "Defensive Difference",
        "ylabel": "Win Probability",
    },
    {# Offensive Difference vs Defensive Difference
        "x": "AdjOE_diff",
        "y": "AdjDE_diff",
        "title": "Offensive Difference vs Defensive Difference",
        "xlabel": "Offensive Difference",
        "ylabel": "Defensive Difference",
    },
    {# EFG% Difference vs Win Probability
        "x": "EFG%_diff",
        "y": "Prob_Team_A_Wins",
        "title": "EFG% Difference vs Win Probability",
        "xlabel": "EFG% Difference",
        "ylabel": "Win Probability",
    },
    {# EFGD% Difference vs Win Probability
        "x": "EFGD%_diff",
        "y": "Prob_Team_A_Wins",
        "title": "EFGD% Difference vs Win Probability",
        "xlabel": "EFGD% Difference",
        "ylabel": "Win Probability",
    },
    {#TOR Difference vs Win Probability
        "x": "TOR_diff",
        "y": "Prob_Team_A_Wins",
        "title": "TOR Difference vs Win Probability",
        "xlabel": "TOR Difference",
        "ylabel": "Win Probability",
    },
    {#TORD Difference vs Win Probability
        "x": "TORD_diff",
        "y": "Prob_Team_A_Wins",
        "title": "TORD Difference vs Win Probability",
        "xlabel": "TORD Difference",
        "ylabel": "Win Probability",
    },
    {#ORB Difference vs Win Probability
        "x": "ORB_diff",
        "y": "Prob_Team_A_Wins",
        "title": "ORB Difference vs Win Probability",
        "xlabel": "ORB Difference",
        "ylabel": "Win Probability",
    },
    {#DRB Difference vs Win Probability
        "x": "DRB_diff",
        "y": "Prob_Team_A_Wins",
        "title": "DRB Difference vs Win Probability",
        "xlabel": "DRB Difference",
        "ylabel": "Win Probability",
    },
    {#Adj Tempo Difference vs Win Probability
        "x": "ADJ T_diff",
        "y": "Prob_Team_A_Wins",
        "title": "Adj T Difference vs Win Probability",
        "xlabel": "Adj T Difference",
        "ylabel": "Win Probability",
    },
    {#WAB Difference vs Win Probability
        "x": "WAB_diff",
        "y": "Prob_Team_A_Wins",
        "title": "WAB Difference vs Win Probability",
        "xlabel": "WAB Difference",
        "ylabel": "Win Probability",
    }
]

data = pd.read_csv('all_probabilities_results_2026.csv')

ENABLED_PLOTS = {10}

for i, p in enumerate(plots):
    if i in ENABLED_PLOTS:
        scatter_plot(
            data[p["x"]],
            data[p["y"]],
            p["title"],
            p["xlabel"],
            p["ylabel"]
        )




