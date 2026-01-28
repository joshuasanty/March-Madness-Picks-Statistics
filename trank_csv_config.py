#To get csv:
#"https://barttorvik.com/trank.php?year=2026&csv=1"

import pandas as pd
import os

raw_data_dir = "raw_trank_data/"

for filename in os.listdir(raw_data_dir):
    if filename.startswith('trank_data_') and filename.endswith('.csv'):
        year = filename.split('_')[-1].split('.')[0]
        input_path = os.path.join(raw_data_dir, filename)
        output_path = f"clean_trank_data/clean_trank_data_{year}.csv"
        print(f"Processing {filename}...")

        df = pd.read_csv(input_path, header=None, engine='python')

        df = df.loc[:, (df != 0).any(axis=0)]
        # print("Columns with all 0s removed!")

        df = df.dropna(axis=1, how='all')
        # print("Empty columns removed!")

        #Remove unknown columns
        columns_to_delete = [20, 21, 24, 27]
        df = df.drop(columns=df.columns[columns_to_delete])
        # 60.1 41.4, 72.7919, 71.8 for Michigan

        #Rename the column labels so they don't skip
        df.columns = range(df.shape[1])

        # print(df.head()) #Sample output
        # print(df.shape[1]) #Number of Columns
        # print(df.loc[0]) #First row


        #Add header row
        main_columns = [
            "Team", "AdjOE", "AdjDE", "BARTHAG",
            "Rec", "Wins", "Games",
            "EFG%", "EFGD%",
            "FTR", "FTRD",
            "TOR", "TORD",
            "ORB", "DRB",
            "ADJ T", "2P%", "2P%D",
            "3P%", "3P%D",
            "3PR", "3PRD",
            "Season", "WAB"
        ]

        new_row_df = pd.DataFrame([main_columns])
        df = pd.concat([new_row_df, df], axis=0)



        #------------------------------------
        #Remove more unnecessary columns
        #------------------------------------
        columns_to_delete = [4, 5, 6, 9, 10,16, 17, 18, 19 ,20, 21]
        df = df.drop(columns=df.columns[columns_to_delete])
        df.columns = range(df.shape[1])

        #Save CSV
        df.to_csv(output_path, index=False, header=False)
        print(f"Saved cleaned data to {output_path}")
print("All files processed!")




