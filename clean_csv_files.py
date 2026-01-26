import pandas as pd

def clean_csv(input_path, output_path):
    # Read CSV, skipping the first row
    df = pd.read_csv(input_path, skiprows=1)

    # Rename columns explicitly
    df = df.rename(columns={
        "Rk": "Rank",
        "W": "Conf_W",
        "L": "Conf_L",
        "W-L%": "Conf_Wpct",
        "W.1": "Overall_W",
        "L.1": "Overall_L",
        "W-L%.1": "Overall_Wpct",
        "Own": "PTS_Own",
        "Opp": "PTS_Opp"
    })

    # Drop Notes column if it exists
    if "Notes" in df.columns:
        df = df.drop(columns=["Notes"])

    # Save cleaned CSV
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    clean_csv(
        input_path="a.txt",
        output_path="team_stats_by_conference/mid-eastern_athletic.csv")