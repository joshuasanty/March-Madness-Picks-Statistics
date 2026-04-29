import pandas as pd

#Removing the FirstD1Season and LastD1Season columns from MTeams.csv
df = pd.read_csv("MTeams.csv")
df = df.drop(columns=df.columns[[2, 3]])
df.to_csv("clean_MTeams.csv", header=True, index = False)

#Removing DayNum,WScore,LScore,WLoc, and NumOT from MNCAATourneyCompactResults.csv
df = pd.read_csv("MNCAATourneyCompactResults.csv")
df = df.drop(columns=df.columns[[1, 3, 5, 6, 7]])
df.to_csv("clean_MNCAATourneyCompactResults.csv", header=True, index = False)




