import pandas as pd
import statsmodels.api as sm
import numpy as np
import random
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

random.seed(42)


def create_diff(df: pd.DataFrame, stats_list: list) -> pd.DataFrame:
    random.seed(42)

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


from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import numpy as np


def train_lasso_logistic_regression_model(
    stats_list: list,
    csv_path: str = "../training_data/training_data.csv",
    season_end: int = 2026,
    tune_C: bool = True,
    default_C: float = 0.02,
    cv: int = 5,
    verbose: bool = False,
):
    df = pd.read_csv(csv_path)
    train_df = df[df["Season"] <= season_end].copy()
    train_data = create_diff(train_df, stats_list)

    feature_cols = [f"{s}_diff" for s in stats_list]

    X = train_data[feature_cols].values
    y = train_data["y"].values

    if tune_C:
        base_model = LogisticRegression(
            penalty='l1',
            solver='saga',
            max_iter=2000,
            random_state=42
        )

        param_grid = {'C': np.linspace(0.001, 0.5, 50)}

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='neg_log_loss',  # better for your problem than ROC-AUC
            n_jobs=-1
        )

        if verbose:
            print("Running GridSearch for C...")

        grid.fit(X, y)

        results = pd.DataFrame(grid.cv_results_)

        # Best model
        best_idx = results['mean_test_score'].idxmax()
        best_score = results.loc[best_idx, 'mean_test_score']
        best_std = results.loc[best_idx, 'std_test_score']

        # 1-standard-error rule (simplest good model)
        threshold = best_score - best_std
        good_models = results[results["mean_test_score"] >= threshold]

        optimal_C = good_models["param_C"].astype(float).min()

        if verbose:
            print(f"Best CV score: {best_score:.4f}")
            print(f"Chosen C (1-SE rule): {optimal_C:.4f}")

    else:
        optimal_C = default_C

    # Fit final model
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=optimal_C,
        max_iter=2000,
        random_state=42
    )

    model.fit(X, y)

    coefs = model.coef_.ravel()

    selected_features = [
        feature_cols[i] for i in range(len(feature_cols)) if coefs[i] != 0
    ]

    if verbose:
        print("\nSelected Features:")
        for f, c in zip(feature_cols, coefs):
            if c != 0:
                print(f"{f}: {c:.4f}")

    return model, selected_features, optimal_C


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