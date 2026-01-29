import pandas as pd
import glob

files = sorted(glob.glob("../tournament_games_data/games_20*.csv"))

df = pd.concat(
    (pd.read_csv(f) for f in files),
    ignore_index = True
)

print(df.shape)
print(df.columns)
print(df.head())

# check missing values
print(df.isna().sum())

# check duplicates
print(df.duplicated().sum())

# verify seasons
print(df["Season"].min(), df["Season"].max())

df.to_csv("training_data.csv", index=False)