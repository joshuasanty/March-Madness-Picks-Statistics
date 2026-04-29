import pandas as pd
import statsmodels.api as sm
import numpy as np

# list of stats to use (keep consistent)
stats = [
    "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
    "ORB", "DRB", "ADJ T", "WAB"
]


def create_diff(df: pd.DataFrame, stats_list: list) -> pd.DataFrame:
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
        csv_path: str = "../training_data/training_data.csv",
        season_end: int = 2025,
        verbose: bool = False,
):
    """
    Train logistic regression using statsmodels.
    Returns: (result, feature_cols)
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Season"] <= season_end].copy()
    train_data = create_diff(train_df, stats)

    feature_cols = [c for c in train_data.columns if c.endswith("_diff")]

    train_data.to_csv("actual_training_data.csv", index=False)

    X = train_data[feature_cols]
    y = train_data["y"]

    # Add intercept
    X = sm.add_constant(X)

    # Fit model
    model = sm.Logit(y, X)
    result = model.fit()

    # Full statistical summary
    print(result.summary())

    # Clean coefficient table
    summary_df = pd.DataFrame({
        "Coefficient": result.params,
        "Std Error": result.bse,
        "p-value": result.pvalues,
        "Odds Ratio": result.params.apply(lambda x: np.exp(x))
    })

    print("\nCoefficient Table:")
    print(summary_df)

    if verbose:
        print("Trained statsmodels Logit. X shape:", X.shape)

    return result, feature_cols


def train_lasso_logistic_regression_model(
        csv_path: str = "training_data/training_data.csv",
        season_end: int = 2023,
        alpha: float = 1.0,
        verbose: bool = False,
):
    """
    L1-penalized logistic regression using statsmodels
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Season"] <= season_end].copy()
    train_data = create_diff(train_df, stats)

    feature_cols = [c for c in train_data.columns if c.endswith("_diff")]

    X = train_data[feature_cols]
    y = train_data["y"]

    X = sm.add_constant(X)

    model = sm.Logit(y, X)

    # L1 regularization
    result = model.fit_regularized(method='l1', alpha=alpha)

    print("\nLasso Coefficients:")
    print(result.params)

    if verbose:
        nonzero = result.params[result.params != 0]
        print("\nSelected Features (non-zero):")
        print(nonzero)

    return result, feature_cols


def build_team_stats(df: pd.DataFrame, season: int) -> pd.DataFrame:
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