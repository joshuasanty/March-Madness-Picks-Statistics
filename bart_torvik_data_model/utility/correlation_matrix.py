#Generating correlation plot of variables

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("../training_data/training_data.csv")
df.drop(columns=["Season", "WTeamID","LTeamID", "TeamID_W", "TeamID_L", "Team_W", "Team_L"], inplace=True)
corr = df.corr().abs()  # absolute value of correlations

# Create a mask for the upper triangle (ignore duplicate pairs)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Find columns with correlation > 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

print("Highly correlated columns (to consider dropping):")
print(to_drop)

plt.matshow(corr)
plt.show()

