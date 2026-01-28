import pandas as pd
import os


processed_files = [f for f in os.listdir("clean_trank_data/")
                  if f.startswith('clean_trank_data_') and f.endswith('.csv')]
years = [f.split('_')[-1].split('.')[0] for f in processed_files]

for year in years:
    print(f"Processing year: {year}")
    try:
        bart = pd.read_csv(f"clean_trank_data/clean_trank_data_{year}.csv")
    except FileNotFoundError:
        print(f"No data found for year {year}. Skipping...")
        continue

    games = pd.read_csv("kaggle_ncaa_data/clean_MNCAATourneyCompactResults.csv")
    teams = pd.read_csv("kaggle_ncaa_data/clean_MTeams.csv")

    #finish modularizing this
    # -----------------------------------------------------------
    # MERGE TEAM NAME AND TEAM ID
    # -----------------------------------------------------------

    # Takes MTeams TeamID and matches it to the trank team names

    bart = bart.merge(
        teams,
        left_on="Team",
        right_on="TeamName",
        how="left"
    )

    cols = list(bart.columns)
    cols.remove('TeamID')
    team_idx = cols.index('Team')
    cols.insert(team_idx, 'TeamID')
    bart = bart[cols]

    bart = bart.drop(columns=bart.columns[-1])

    # -----------------------------------------------------------
    # MERGE TEAM STATS AND TOURNAMENT RESULTS
    # -----------------------------------------------------------

    #Winners:
    games = games.merge(
        bart,
        left_on=["Season", "WTeamID"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("", "_W")
    )

    games = games.merge(
        bart,
        left_on=["Season", "LTeamID"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("_W", "_L")
    )


    # ----------------------------------------------------
    # Debugging
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(bart.head(1))
    print(games.head(1))

    #Delete data rows that are empty
    games = games.dropna(subset=["BARTHAG_W", "BARTHAG_L"])

    output_path = f"tournament_games_data/games_{year}.csv"
    games.to_csv(output_path, index=False, header=True)
    print(f"Successfully processed and saved data for {year} to {output_path}")

