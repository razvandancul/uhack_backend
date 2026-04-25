import pandas as pd

df = pd.read_csv("players.csv")
print(f"Rânduri: {len(df)}")
print(f"Coloane: {df.columns.tolist()}")
print(df.head(3))