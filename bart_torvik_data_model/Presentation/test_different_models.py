from model_train_for_presentation import train_logistic_regression_model, train_lasso_logistic_regression_model
import itertools
import statsmodels.api as sm
import pandas as pd

doALL = True
doREDUCED = False
DO_LASSO = False
DO_BEST_SUBSET = False
DO_FORWARD_STEPWISE = False

if doALL:
    all_stats = [
        "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
        "ORB", "DRB", "ADJ T", "WAB"
    ]

    # Full Models
    log_model_all, log_features_all, log_summary_df_all = train_logistic_regression_model(all_stats)
    lasso_model_all, lasso_features_all, best_C = train_lasso_logistic_regression_model(all_stats)
    print(log_model_all.summary())
    # print(log_summary_df_all)

if doREDUCED:
    reduced_stats = ["AdjOE", "AdjDE", "DRB", "WAB"]
    #Reduce Logit
    logit_model_reduced, log_features_reduced, log_summary_reduced = train_logistic_regression_model(reduced_stats)
    print(logit_model_reduced.summary())

# print(lasso_model_all.summary())

if DO_LASSO:
    default_stats = [
        "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
        "ORB", "DRB", "ADJ T", "WAB"
    ]
    lasso_model, selected_features, best_C = train_lasso_logistic_regression_model(
        default_stats,
        verbose=True
    )

    print("Final selected features:", selected_features)
    selected_stats = [f.replace("_diff", "") for f in selected_features]

    logit_model, _, summary = train_logistic_regression_model(selected_stats)
    print(logit_model.summary())

if DO_BEST_SUBSET:
    train_df = pd.read_csv("actual_training_data.csv")
    X_full = train_df.drop(columns=["y"])
    y = train_df["y"]

    best_aic = float("inf")
    best_model = None
    best_features = None

    features = X_full.columns.tolist()

    for k in range(1, len(features)+1):
        for subset in itertools.combinations(features, k):
            try:
                X_subset = sm.add_constant(X_full[list(subset)])
                model = sm.Logit(y, X_subset).fit(disp=0)

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_features = subset
                    best_model = model
            except:
                continue

    print("Best subset:", best_features)
    print("Best AIC:", best_aic)

if DO_FORWARD_STEPWISE:
    def forward_stepwise(X, y):
        import statsmodels.api as sm

        remaining = list(X.columns)
        selected = []
        current_aic = float("inf")

        while remaining:
            aic_with_candidates = []

            for candidate in remaining:
                features = selected + [candidate]
                X_model = sm.add_constant(X[features])
                model = sm.Logit(y, X_model).fit(disp=0)
                aic_with_candidates.append((model.aic, candidate, model))

            aic_with_candidates.sort()
            best_new_aic, best_candidate, best_model = aic_with_candidates[0]

            if best_new_aic < current_aic:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_aic = best_new_aic
            else:
                break

        return selected, best_model


    train_df = pd.read_csv("actual_training_data.csv")
    X_full = train_df.drop(columns=["y"])
    y = train_df["y"]
    selected_features, stepwise_model = forward_stepwise(X_full, y)
    print("Selected:", selected_features)
    print(stepwise_model.summary())