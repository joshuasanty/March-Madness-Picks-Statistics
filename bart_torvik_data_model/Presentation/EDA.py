#----------------------------
# EDA
# Get Mean, Std Dev, Min, and Max of each stat
#----------------------------

import pandas as pd
import glob


combined_df = pd.concat(
    [pd.read_csv(f) for f in glob.glob("../all_probability_data/20*.csv") if "2020" not in f],
    ignore_index=True
)
print(combined_df.shape)
stats = ["AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD", "ORB", "DRB", "ADJ T", "WAB"]

for stat in stats:
    print(f"Mean {stat}: {combined_df[stat].mean()}")
    print(f"Std Dev {stat}: {combined_df[stat].std()}")
    print(f"Min {stat}: {combined_df[stat].min()}")
    print(f"Max {stat}: {combined_df[stat].max()}")
