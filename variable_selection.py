import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("training_data/training_data.csv")

# Drop identifiers / non-numeric columns
df.drop(columns=["Season","TeamID_W","Team_W","TeamID_L","Team_L","WTeamID","LTeamID"], inplace=True)

# --- Mirror dataset to include both win/loss perspectives ---
df_mirror = df.copy()

# Swap W/L stats for mirrored dataset
for col in df.columns:
    if col.endswith("_W"):
        df_mirror[col.replace("_W","_L")] = df[col]
    elif col.endswith("_L"):
        df_mirror[col.replace("_L","_W")] = df[col]
df_mirror = df_mirror[df.columns]  # ensure same column order

# Add target: original wins = 1, mirrored wins = 0
df['target'] = 1
df_mirror['target'] = 0

# Combine original + mirrored
df_full = pd.concat([df, df_mirror], ignore_index=True)

# --- Prepare X/y for mutual information ---
y = df_full['target']
X = df_full.drop(columns=['target'])

# Encode target if necessary
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Compute mutual information scores
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Display top features
print("Feature ranking by mutual information:")
print(mi_series)

# Optional: select top 10 features
top_features = mi_series.head(10).index
print("\nTop 10 features to consider:")
print(top_features)