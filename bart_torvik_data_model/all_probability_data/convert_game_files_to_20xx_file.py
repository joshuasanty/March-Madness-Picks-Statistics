#This code takes the game files from tournament_games_data/ and turns them into 20xx.csv files
#This is necessary for any new season when doing cross validation

#Have to go in and manually delete duplicates

import pandas as pd
from pathlib import Path

INPUT_DIR = Path("../tournament_games_data")
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

stat_cols = [
    "AdjOE", "AdjDE", "BARTHAG", "EFG%", "EFGD%",
    "TOR", "TORD", "ORB", "DRB", "ADJ T", "WAB"
]

def convert_games_file(input_csv: Path, output_csv: Path):
    if not input_csv.exists():
        print(f"Skipping {input_csv} (file not found)")
        return

    df = pd.read_csv(input_csv)
    rows = []

    for _, r in df.iterrows():
        season = int(r["Season"])

        winner_row = {"Team": r["Team_W"], "Season": season}
        loser_row = {"Team": r["Team_L"], "Season": season}

        for col in stat_cols:
            winner_row[col] = r[f"{col}_W"]
            loser_row[col] = r[f"{col}_L"]

        rows.append(winner_row)
        rows.append(loser_row)

    out_df = pd.DataFrame(rows)

    # Remove duplicate teams (keep first occurrence)
    out_df = out_df.drop_duplicates(subset=["Team"])

    # Ensure correct column order
    out_df = out_df[["Team"] + stat_cols + ["Season"]]

    out_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


# Loop through years, skipping missing ones
for year in range(2010, 2027):
    input_csv = INPUT_DIR / f"games_{year}.csv"
    output_csv = OUTPUT_DIR / f"{year}.csv"
    convert_games_file(input_csv, output_csv)