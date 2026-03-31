import pandas as pd

from model_training import (
    train_logistic_regression_model,
    train_lasso_logistic_regression_model,
    stats,
)

# ------------------------------
# CONFIG
# ------------------------------
TRAINING_CSV = "training_data/training_data.csv"
BRACKET_FILE = "tournament_simulation/ordered_games_2026.csv"
USE_LOGISTIC = True
USE_LASSO = False
SAVE_PREDICTIONS_TO = "all_probabilities_2026.csv"

bracket_df = pd.read_csv(BRACKET_FILE)

#need to change this if doing cross validation
# -------------------------
if "Season" in bracket_df.columns and len(bracket_df) > 0:
    test_season = int(bracket_df["Season"].iloc[0])
else:
    test_season = 2026

train_season_end = test_season - 1
#-------------------------

#Train Model:
if USE_LOGISTIC:
    model, feature_cols = train_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end,
        verbose=False
    )
else:
    model = train_lasso_logistic_regression_model(
        csv_path=TRAINING_CSV,
        season_end=train_season_end,
        C=1.0,
        verbose=False
    )
#debugging - get the coefficients of the model
print(model.coef_)
print(feature_cols)

# ------------------------------
# PREDICTIONS
# ------------------------------

#Predict every single possible combination
for i in range(len(bracket_df)):
    bracket_df.loc[i, "Prob"] = model.predict_proba(bracket_df.loc[i, feature_cols])[:, 1]








