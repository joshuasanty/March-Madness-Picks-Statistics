import pandas as pd

#Removing the FirstD1Season and LastD1Season columns from MTeams.csv
df = pd.read_csv("MTeams.csv")
df = df.drop(columns=df.columns[[2, 3]])
df.to_csv("clean_MTeams.csv", header=True, index = False)


