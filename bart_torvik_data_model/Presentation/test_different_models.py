from model_train_for_presentation import train_logistic_regression_model, train_lasso_logistic_regression_model

all_stats = [
    "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
    "ORB", "DRB", "ADJ T", "WAB"
]

reduced_stats = ["AdjOE", "AdjDE", "DRB", "WAB"]

#Full Models
log_model_all, log_features_all, log_summary_df_all = train_logistic_regression_model(all_stats)
lasso_model_all, lasso_features_all = train_lasso_logistic_regression_model(all_stats)

#Reduce Logit
logit_model_reduced, log_features_reduced, log_summary_reduced = train_logistic_regression_model(reduced_stats)

print(log_model_all.summary())
# print(log_summary_df_all)

print(logit_model_reduced.summary())
# print(lasso_model_all.summary())