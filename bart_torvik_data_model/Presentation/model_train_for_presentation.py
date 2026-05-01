import pandas as pd
import statsmodels.api as sm
import numpy as np
import random

random.seed(42)


def create_diff(df: pd.DataFrame, stats_list: list) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if random.random() < 0.5:
            # Winner as TeamA
            row = {f"{s}_diff": r[f"{s}_W"] - r[f"{s}_L"] for s in stats_list}
            row["y"] = 1
        else:
            # Loser as TeamA
            row = {f"{s}_diff": r[f"{s}_L"] - r[f"{s}_W"] for s in stats_list}
            row["y"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def train_logistic_regression_model(
    stats_list: list,
    csv_path: str = "../training_data/training_data.csv",
    season_end: int = 2026,
    verbose: bool = False,
):
    """
    Train logistic regression using statsmodels.
    Returns: (result, feature_cols)
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Season"] <= season_end].copy()
    train_data = create_diff(train_df, stats_list)

    feature_cols = [f"{s}_diff" for s in stats_list]

    train_data.to_csv("actual_training_data.csv", index=False)

    X = train_data[feature_cols]
    y = train_data["y"]

    # Add intercept
    X = sm.add_constant(X)

    # Fit model
    model = sm.Logit(y, X)
    result = model.fit()

    # Full statistical summary
    # print(result.summary())

    # Clean coefficient table
    summary_df = pd.DataFrame({
        "Coefficient": result.params,
        "Std Error": result.bse,
        "p-value": result.pvalues,
        "Odds Ratio": np.exp(result.params)
    })

    # print("\nCoefficient Table:")
    # print(summary_df)

    if verbose:
        print("Trained statsmodels Logit. X shape:", X.shape)

    return result, feature_cols, summary_df


def train_lasso_logistic_regression_model(
    stats_list: list,
    csv_path: str = "../training_data/training_data.csv",
    season_end: int = 2026,
    alpha: float = 1.0,
    verbose: bool = False,
):
    """
    L1-penalized logistic regression using statsmodels
    Returns: (result, feature_cols)
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["Season"] <= season_end].copy()
    train_data = create_diff(train_df, stats_list)

    feature_cols = [f"{s}_diff" for s in stats_list]

    X = train_data[feature_cols]
    y = train_data["y"]

    X = sm.add_constant(X)

    model = sm.Logit(y, X)

    # L1 regularization
    result = model.fit_regularized(method='l1', alpha=alpha)

    # print("\nLasso Coefficients:")
    # print(result.params)

    if verbose:
        nonzero = result.params[result.params != 0]
        print("\nSelected Features (non-zero):")
        print(nonzero)

    return result, feature_cols


def build_team_stats(df: pd.DataFrame, season: int, stats_list: list) -> pd.DataFrame:
    winners = df[df["Season"] == season][
        ["Team_W"] + [f"{s}_W" for s in stats_list]
    ].rename(columns=lambda c: c.replace("_W", "") if "_W" in c else c)

    losers = df[df["Season"] == season][
        ["Team_L"] + [f"{s}_L" for s in stats_list]
    ].rename(columns=lambda c: c.replace("_L", "") if "_L" in c else c)

    winners = winners.rename(columns={"Team_W": "Team"})
    losers = losers.rename(columns={"Team_L": "Team"})

    team_stats = pd.concat([winners, losers]).drop_duplicates("Team").set_index("Team")
    return team_stats


if __name__ == "__main__":
    default_stats = [
        "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
        "ORB", "DRB", "ADJ T", "WAB"
    ]

    train_logistic_regression_model(default_stats)