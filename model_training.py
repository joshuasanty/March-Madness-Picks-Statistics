import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

df = pd.read_csv("training_data/training_data.csv")

train_df = df[df["Season"] <= 2023].copy()
test_df = df[df["Season"] == 2024].copy()

stats = [
    "AdjOE", "AdjDE", "BARTHAG",
    "EFG%", "EFGD%", "TOR", "TORD",
    "ORB", "DRB", "ADJ T", "WAB"
]


#Create diff
def create_diff(df, stats):
    rows = []

    for _, r in df.iterrows():

        #In the data, the winner is always listed first!

        #Team A Winner
        row_win = {}
        for s in stats:
            row_win[f"{s}_diff"] = r[f"{s}_W"] - r[f"{s}_L"]
        row_win["y"] = 1
        rows.append(row_win)

        row_win["TeamA"] = r["Team_W"]
        row_win["TeamB"] = r["Team_L"]
        rows.append(row_win)

        #Team A Loser
        row_lose = {}
        for s in stats:
            row_lose[f"{s}_diff"] = r[f"{s}_L"]-r[f"{s}_W"]
        row_lose["y"] = 0
        rows.append(row_lose)

        row_lose["TeamA"] = r["Team_L"]
        row_lose["TeamB"] = r["Team_W"]
        rows.append(row_lose)


    return pd.DataFrame(rows)



train_data = create_diff(train_df, stats)
test_data = create_diff(test_df, stats)

feature_cols = [c for c in train_data.columns if c.endswith("_diff")]

x_train = train_data[feature_cols]
y_train = train_data["y"]

x_test = test_data[feature_cols]
y_test = test_data["y"]

#------------- Run the model
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs"
)

model.fit(x_train, y_train)

probs = model.predict_proba(x_test)[:, 1]
output = test_data[["TeamA", "TeamB"]].copy()
output["P_TeamA_Wins"] = probs
output["P_TeamB_Wins"] = 1 - probs

output = output.join(test_data["y"])
output = output[output["y"] == 1]


for _, r in output.iterrows():
    print(
        f"{r['TeamA']} vs {r['TeamB']} — "
        f"{r['TeamA']}: {r['P_TeamA_Wins']:.1%}, "
        f"{r['TeamB']}: {r['P_TeamB_Wins']:.1%}"
    )
preds = (probs > 0.5).astype(int)


# print(probs)
# print(preds)

print("Accuracy: ", accuracy_score(y_test, preds))
print("Log Loss: ", log_loss(y_test, probs))


#----------- Test example for UConn v Purdue
# test_df_example = df.iloc[[-1]].copy()
# print(test_df_example)
# test_data_example = create_diff(test_df_example, stats)
# # print(test_data_example.to_string())
#
# x_test_example = test_data_example.drop(columns="y")
# y_test_example = test_data_example["y"]
#
# probs = model.predict_proba(x_test_example)[:, 1]
# preds = (probs > 0.5).astype(int)

# print(probs)
# print(preds)
#
# print("Accuracy: ", accuracy_score(y_test_example, preds))
# print("Log Loss: ", log_loss(y_test_example, probs))
#
# print("Mean square error: ", mean_squared_error(y_test_example, probs))

