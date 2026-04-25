import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Verificăm câți jucători au marketValue și contract
players = db.collection("players").stream()

total = 0
has_value = 0
has_contract = 0

for p in players:
    data = p.to_dict()
    total += 1
    if data.get("marketValueEur"):
        has_value += 1
    if data.get("contractExpiry"):
        has_contract += 1

print(f"Total jucători: {total}")
print(f"Cu valoare de piață: {has_value}")
print(f"Cu contract: {has_contract}")