import pandas as pd


bart = pd.read_csv("clean_trank_data/clean_trank_data_2015.csv")
teams = pd.read_csv("kaggle_ncaa_data/clean_MTeams.csv")

bart = bart.merge(
    teams,
    left_on = "Team",
    right_on = "TeamName",
    how="left"
)

cols = list(bart.columns)
cols.remove('TeamID')
team_idx = cols.index('Team')
cols.insert(team_idx, 'TeamID')
bart = bart[cols]

bart = bart.drop(columns=bart.columns[-1])

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(bart.head(1))