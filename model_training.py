import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("training_data/training_data.csv")

train_df = df[df["Season"] <= 2022].copy()
test_df = df[df["Season"] >= 2023].copy()

stats = [
    "AdjOE", "AdjDE", "BARTHAG",
    "EFG%", "EFGD%", "TOR", "TORD",
    "ORB", "DRB", "ADJ T", "WAB"
]

#Create diff
def create_diff(df, stats):
    rows = []

    for _, r in df.iterrows():
        #Team A Winner
        row_win = {}
        for s in stats:
            row_win[f"{s}_diff"] = r[f"{s}_W"] - r[f"{s}_L"]
        row_win["y"] = 1
        rows.append(row_win)

        #Team A Loser
        row_lose = {}
        for s in stats:
            row_lose[f"{s}_diff"] = r[f"{s}_L"]-r[f"{s}_W"]
        row_lose["y"] = 0
        rows.append(row_lose)

    return pd.DataFrame(rows)


train_data = create_diff(train_df, stats)
test_data = create_diff(train_df, stats)

x_train = train_data.drop(columns="y")
y_train = train_data["y"]

x_test = test_data.drop(columns="y")
y_test = test_data["y"]

#------------- Run the model
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs"
)

model.fit(x_train, y_train)

probs = model.predict_proba(x_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("Accuracy: ", accuracy_score(y_test, preds))
print("Log Loss: ", log_loss(y_test, probs))
