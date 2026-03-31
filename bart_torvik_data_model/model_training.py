# This file trains all the models
# (Currently Logistic Regression and Lasso Logistic Regression)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# list of stats to use (keep consistent)
stats = [
    "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
    "ORB", "DRB", "ADJ T", "WAB"
]


# Taken out:


def create_diff(df: pd.DataFrame, stats_list: list) -> pd.DataFrame:
    """
    From a dataframe where each row is a game with winner (suffix _W) and loser (suffix _L)
    produce a dataframe of differences with one row per ordering:
      - TeamA = winner, TeamB = loser, y = 1 (win)
      - TeamA = loser,  TeamB = winner, y = 0 (loss)
    """
    rows = []
    for _, r in df.iterrows():
        # Winner as TeamA (label 1)
        win_row = {f"{s}_diff": r[f"{s}_W"] - r[f"{s}_L"] for s in stats_list}
        win_row["y"] = 1
        win_row["TeamA"] = r["Team_W"]
        win_row["TeamB"] = r["Team_L"]
        rows.append(win_row)

        # Loser as TeamA (label 0)
        lose_row = {f"{s}_diff": r[f"{s}_L"] - r[f"{s}_W"] for s in stats_list}
        lose_row["y"] = 0
        lose_row["TeamA"] = r["Team_L"]
        lose_row["TeamB"] = r["Team_W"]
        rows.append(lose_row)

    return pd.DataFrame(rows)


def train_logistic_regression_model(
        csv_path: str = "training_data/training_data.csv",
        season_end: int = 2024,
        verbose: bool = False,
):
    """
    Train a plain logistic regression model on seasons <= season_end.
    Returns: (model, feature_cols)
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Season"] <= season_end].copy()
    train_data = create_diff(train_df, stats)

    feature_cols = [c for c in train_data.columns if c.endswith("_diff")]

    train_data.to_csv("actual_training_data.csv", index=False) #This is the actual training data
    X = train_data[feature_cols]
    y = train_data["y"]

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        l1_ratio=0.0  # equivalent to L2
    )
    model.fit(X, y)

    print("\nCoefficients:")
    for feature, coef in zip(feature_cols, model.coef_[0]):
        print(f"{feature}: {coef:.6f}")

    print(f"intercept: {model.intercept_[0]:.6f}")

    if verbose:
        print("Trained LogisticRegression. X shape:", X.shape)

    return model, feature_cols


def train_lasso_logistic_regression_model(
        csv_path: str = "training_data/training_data.csv",
        season_end: int = 2023,
        C: float = 1.0,
        verbose: bool = False,
):
    """
    Train an L1-penalized logistic regression model (Lasso logistic).
    Returns: (model, feature_cols)
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Season"] <= season_end].copy()
    train_data = create_diff(train_df, stats)

    feature_cols = [c for c in train_data.columns if c.endswith("_diff")]

    X = train_data[feature_cols]
    y = train_data["y"]

    # encode if object
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    model = LogisticRegression(
        solver="saga",  # required for l1_ratio
        l1_ratio=1.0,  # pure L1 (lasso)
        max_iter=1000,
        C=1.0,
    )
    model.fit(X, y)
    print("\nCoefficients:")
    for feature, coef in zip(feature_cols, model.coef_[0]):
        print(f"{feature}: {coef:.6f}")

    print(f"intercept: {model.intercept_[0]:.6f}")
    if verbose:
        # show selected features
        coefs = pd.Series(model.coef_[0], index=feature_cols)
        selected = coefs[coefs != 0].sort_values(ascending=False)
        print("Lasso selected features (non-zero):")
        print(selected)

    return model, feature_cols


def build_team_stats(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by Team name with raw stats for the given season.
    Not called by the training functions by default; provided for downstream use.
    """
    winners = df[df["Season"] == season][
        ["Team_W"] + [f"{s}_W" for s in stats]
        ].rename(columns=lambda c: c.replace("_W", "") if "_W" in c else c)

    losers = df[df["Season"] == season][
        ["Team_L"] + [f"{s}_L" for s in stats]
        ].rename(columns=lambda c: c.replace("_L", "") if "_L" in c else c)

    winners = winners.rename(columns={"Team_W": "Team"})
    losers = losers.rename(columns={"Team_L": "Team"})

    team_stats = pd.concat([winners, losers]).drop_duplicates("Team").set_index("Team")
    return team_stats

if __name__ == "__main__":
    train_logistic_regression_model()
    # train_lasso_logistic_regression_model()