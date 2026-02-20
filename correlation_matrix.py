#Generating correlation plot of variables

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("training_data/training_data.csv")
df.drop(columns=["Season", "WTeamID","LTeamID", "TeamID_W", "TeamID_L", "Team_W", "Team_L"], inplace=True)
corr = df.corr()
plt.matshow(corr)

corr.style.background_gradient(cmap='coolwarm').set_precision(2)

plt.show()

