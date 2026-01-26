#To get csv:
#"https://barttorvik.com/trank.php?year=2024&csv=1"

import pandas as pd

df = pd.read_csv("trank_data.csv", header=None, engine='python')

df_clean = df.dropna(axis=1, how='all')
print("Empty columns removed!")

df_clean = df_clean.loc[:, (df_clean != 0).any(axis=0)]
print("Columns with all 0s removed!")

# #Add header row to see which columns of 0 to remove
# # print(df.shape[1]) Number of Columns
# header_row = list(range(1, df.shape[1] + 1))
# new_row_df = pd.DataFrame([header_row])
#
# #Add header row
# df_clean = pd.concat([new_row_df, df_clean], axis=0)

#Delete columns with all 0s




df_clean.to_csv("clean_trank_data.csv", index=False, header=False)

main_columns = [
    "Team", "AdjOE", "AdjDE", "BARTHAG",
    "Rec", "Wins", "Games",
    "EFG%", "EFGD%",
    "FTR", "FTRD",
    "TOR", "TORD",
    "ORB", "DRB",
    "ADJ T", "2P%", "2P%D",
    "3P%", "3P%D",
]



