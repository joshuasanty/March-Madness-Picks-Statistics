from model_train_for_presentation import train_logistic_regression_model, train_lasso_logistic_regression_model

my_stats = [
    "AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD",
    "ORB", "DRB", "ADJ T", "WAB"
]

logit_model, features, summary_df = train_logistic_regression_model(my_stats)
lasso_model, features = train_lasso_logistic_regression_model(my_stats)

print(logit_model.summary())
print(summary_df)
print(lasso_model.summary())