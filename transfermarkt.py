import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Descărcăm direct din GitHub
url = "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data/players.csv"
df = pd.read_csv(url)

print(df.columns.tolist())
print(df.head(3))