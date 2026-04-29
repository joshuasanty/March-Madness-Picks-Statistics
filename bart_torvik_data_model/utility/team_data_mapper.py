import os
import pandas as pd

def normalize(name):
    return (
        str(name).lower()
        .replace(".", "")
        .replace("&", "and")
        .strip()
    )

# Optional: manually fix known mismatches (add to this over time)
name_map = {
    # Example entries (edit as needed)
    # "ole miss": "mississippi",
    # "uconn": "connecticut",
    "north dakota st": "n dakota st",
    "prairie view aandm": "prairie view",
    "saint mary's":"st mary's ca",
    "liu":"liu brooklyn",
    "kennesaw st":"kennesaw",
    "queens":"queens nc",
    "saint louis": "st louis",
    "saint francis":"st francis pa",
    "mount st mary's":"mt st mary's",
    "american":"american univ",
    "siu edwardsville":"siue",
    "nebraska omaha":"ne omaha",
    "grambling st":"grambling",
    "south dakota st":"s dakota st",
    "saint peter's":"st peter's",
    "charleston":"col charleston",
    "western kentucky":"wku",
    "florida atlantic": "fl atlantic",
    "grambling st": "grambling",
    "texas aandm corpus chris": "tam c christi",
    "texas southern": "tx southern",
    "fairleigh dickinson": "f dickinson",
    "northern kentucky": "n kentucky",
    "kent st": "kent",
    "southeast missouri st": "se missouri st",
    "cal st fullerton": "cs fullerton",
    "loyola chicago": "loyola-chicago",
    "abilene christian": "abilene chr",
    "eastern washington": "e washington",
    "north carolina central": "nc central",
    "stephen f austin": "sf austin",
    "middle tennessee": "mtsu",
    "east tennessee st": "etsu",
    "florida gulf coast": "fgcu",
    "southern": "southern univ",
    "little rock": "ark little rock",
    "cal st bakersfield": "cs bakersfield",
    "saint joseph's": "st joseph's pa",
    "green bay": "wi green bay",
    "albany": "suny albany",
    "western michigan": "w michigan",
    "coastal carolina": "coastal car",
    "milwaukee": "wi milwaukee",
    "eastern kentucky": "e kentucky",
    "george washington": "g washington",
    "north carolina aandt": "nc aandt",
    "northwestern st": "northwestern la",
    "mississippi valley st": "ms valley st",
    "utsa": "ut san antonio",
    "detroit mercy": "detroit",
    "northern colorado": "n colorado",
    "boston university": "boston univ",
    "arkansas pine bluff": "ark pine bluff",
    "": "",
    "": "",
    "": "",

}

processed_files = [
    f for f in os.listdir("../clean_trank_data/")
    if f.startswith('clean_trank_data_') and f.endswith('.csv')
]

years = [f.split('_')[-1].split('.')[0] for f in processed_files]

for year in years:
    print(f"\nProcessing year: {year}")

    try:
        bart = pd.read_csv(f"../clean_trank_data/clean_trank_data_{year}.csv")
    except FileNotFoundError:
        print(f"No data found for year {year}. Skipping...")
        continue
    # print(bart[(bart["Team"] == "Florida")].to_string())
    games = pd.read_csv("../kaggle_ncaa_data/clean_MNCAATourneyCompactResults.csv")
    teams = pd.read_csv("../kaggle_ncaa_data/clean_MTeams.csv")

    # Filter to only this season
    games = games[games["Season"] == int(year)].copy()
    print("Tournament games in season:", len(games))

    # -----------------------------
    # Normalize names
    # -----------------------------
    bart["Team_norm"] = bart["Team"].apply(normalize)
    teams["TeamName_norm"] = teams["TeamName"].apply(normalize)

    # Apply manual fixes (normalized)
    bart["Team_norm"] = bart["Team_norm"].replace(name_map)

    # -----------------------------
    # Merge bart with team IDs
    # -----------------------------
    bart = bart.merge(
        teams[["TeamID", "TeamName_norm"]],
        left_on="Team_norm",
        right_on="TeamName_norm",
        how="left"
    )
    # print(bart[(bart["Team"] == "Florida")].to_string())

    # Debug: unmatched trank teams
    unmatched = bart[bart["TeamID"].isna()]
    # if len(unmatched) > 0:
    #     print("\nUnmatched trank teams after merge:")
    #     print(unmatched[["Team", "Team_norm"]].drop_duplicates())

    # Drop helper column safely
    bart = bart.drop(columns=["TeamName_norm"])

    # Reorder columns to put TeamID next to Team
    cols = list(bart.columns)
    cols.remove("TeamID")
    team_idx = cols.index("Team")
    cols.insert(team_idx, "TeamID")
    bart = bart[cols]

    # -----------------------------
    # Merge into games
    # -----------------------------
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

    # -----------------------------
    # Debug missing stats
    # -----------------------------
    bad_games = games[
        games["BARTHAG_W"].isna() | games["BARTHAG_L"].isna()
    ]

    print("Rows after merges:", len(games))
    print("Missing winner stats:", games["BARTHAG_W"].isna().sum())
    print("Missing loser stats:", games["BARTHAG_L"].isna().sum())

    if len(bad_games) > 0:
        print("\nProblem TeamIDs (winners):", bad_games["WTeamID"].unique())
        print("Problem TeamIDs (losers):", bad_games["LTeamID"].unique())


    # -----------------------------
    # Drop incomplete rows
    # -----------------------------
    games = games.dropna(subset=["BARTHAG_W", "BARTHAG_L"])

    print("Rows after dropna:", len(games))

    # -----------------------------
    # Save output
    # -----------------------------
    output_path = f"../tournament_games_data/games_{year}.csv"
    games = games.drop(columns=[col for col in games.columns if "norm" in col.lower()])

    games.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")