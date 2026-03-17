# This file is used to train the model using the training data. Currently
# only logistic regression is implemented, but other models can be added.

import pandas as pd
from sklearn.linear_model import LogisticRegression

stats = [ "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
    "ORB", "DRB", "ADJ T", "WAB"
]


# Extra stats taken out:  "BARTHAG",

# Create diff
def create_diff(df, stats):
    rows = []

    for _, r in df.iterrows():

        # In the data, the winner is always listed first!

        # Team A Winner
        row_win = {}
        for s in stats:
            row_win[f"{s}_diff"] = r[f"{s}_W"] - r[f"{s}_L"]
        row_win["y"] = 1
        rows.append(row_win)

        row_win["TeamA"] = r["Team_W"]
        row_win["TeamB"] = r["Team_L"]
        rows.append(row_win)

        # Team A Loser
        row_lose = {}
        for s in stats:
            row_lose[f"{s}_diff"] = r[f"{s}_L"] - r[f"{s}_W"]
        row_lose["y"] = 0
        rows.append(row_lose)

        row_lose["TeamA"] = r["Team_L"]
        row_lose["TeamB"] = r["Team_W"]
        rows.append(row_lose)

    return pd.DataFrame(rows)


#-----------------------------------
#------------ MODELS ---------------
#-----------------------------------


#-----------------------------------
# --- LOGISTIC REGRESSION ---
#-----------------------------------

def train_logistic_regression_model():
    df = pd.read_csv(
        "C:/Users/joshu/PycharmProjects/PythonProject/March-Madness-Picks-Statistics/training_data/training_data.csv")

    train_df = df[df["Season"] <= 2023].copy()

    train_data = create_diff(train_df, stats)

    feature_cols = [c for c in train_data.columns if c.endswith("_diff")]

    X = train_data[feature_cols]
    y = train_data["y"]

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X, y)

    team_stats = build_team_stats(df, season=2024)

    return model, feature_cols, team_stats

#-----------------------------------
# --- LASSO LOGISTIC REGRESSION ---
#-----------------------------------
def train_lasso_logistic_regression_model():
    df = pd.read_csv(
        "C:/Users/joshu/PycharmProjects/PythonProject/March-Madness-Picks-Statistics/training_data/training_data.csv")

    train_df = df[df["Season"] <= 2023].copy()

    train_data = create_diff(train_df, stats)

    feature_cols = [c for c in train_data.columns if c.endswith("_diff")]

    X = train_data[feature_cols]
    y = train_data["y"]

    # Encode target if needed
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)

    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        # l1_ratio=1.0,
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    model.fit(X, y)

    #debug:
    print("Unique classes in y:", y.unique())
    print("X shape:", X.shape)
    print("Model fitted. Does it have coefficients?", hasattr(model, "coef_"))
    #Getting variables chosen:
    coefs = pd.Series(model.coef_[0], index=feature_cols)
    selected_features = coefs[coefs != 0].sort_values(ascending=False)
    print("Features selected by Lasso (non-zero coefficients):")
    print(selected_features)

    team_stats = build_team_stats(df, season=2024)

    return model, feature_cols, team_stats

def build_team_stats(df, season):
    """
    Returns a DataFrame indexed by Team name with raw stats.
    Uses winner & loser rows to ensure all teams included.
    """
    winners = df[df["Season"] == season][
        ["Team_W"] + [f"{s}_W" for s in stats]
        ].rename(columns=lambda c: c.replace("_W", "") if "_W" in c else c)

    losers = df[df["Season"] == season][
        ["Team_L"] + [f"{s}_L" for s in stats]
        ].rename(columns=lambda c: c.replace("_L", "") if "_L" in c else c)

    winners = winners.rename(columns={"Team_W": "Team"})
    losers = losers.rename(columns={"Team_L": "Team"})

    team_stats = (
        pd.concat([winners, losers])
        .drop_duplicates("Team")
        .set_index("Team")
    )

    return team_stats
