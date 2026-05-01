# ----------------------------
# EDA
# Get Mean, Std Dev, Min, and Max of each stat
# ----------------------------
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

doEDA = False
doCORR = True

combined_df = pd.concat(
    [pd.read_csv(f) for f in glob.glob("../all_probability_data/20*.csv") if "2020" not in f],
    ignore_index=True
)
# print(combined_df.shape)
stats = ["AdjOE", "AdjDE", "EFG%", "EFGD%", "TOR", "TORD", "ORB", "DRB", "ADJ T", "WAB"]

if doEDA:
    for stat in stats:
        print(f"Mean {stat}: {combined_df[stat].mean()}")
        print(f"Std Dev {stat}: {combined_df[stat].std()}")
        print(f"Min {stat}: {combined_df[stat].min()}")
        print(f"Max {stat}: {combined_df[stat].max()}")


#--------------------
# Correlation
#--------------------

if doCORR:
    corr_df = pd.read_csv("actual_training_data.csv")
    corr_df = corr_df.drop(columns=["y"])
    print(corr_df)
    corr = corr_df.corr().abs()  # absolute value of correlations

    # Create a mask for the upper triangle (ignore duplicate pairs)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Find columns with correlation > 0.8
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

    print("Highly correlated columns (to consider dropping):")
    print(to_drop)

    #VIF:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Compute VIF
    vif = pd.DataFrame()
    vif["Feature"] = corr_df.columns
    vif["VIF"] = [variance_inflation_factor(corr_df.values, i) for i in range(corr_df.shape[1])]

    print(vif.sort_values(by="VIF", ascending=False))

    # x = corr_df.drop(columns=["WAB"])

    plt.matshow(corr)
    plt.colorbar()
    plt.title("Correlation Matrix")
    #put the title on the bottom
    plt.title("Correlation Matrix", loc="center", y=-0.1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
