import pandas as pd
import firebase_admin
from firebase_admin import firestore

# Uses Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS locally).
firebase_admin.initialize_app()
db = firestore.client()

# Citim CSV
df = pd.read_csv("players.csv")
print(f"✅ {len(df)} jucători în CSV")

# Citim jucătorii din Firebase
print("📡 Citim jucătorii din Firebase...")
players_docs = db.collection("players").stream()
players = [p.to_dict() for p in players_docs]
print(f"✅ {len(players)} jucători în Firebase")

enriched = 0
not_found = 0

for player in players:
    short_name = player.get("shortName", "")
    first_name = player.get("firstName", "").strip()
    last_name = player.get("lastName", "").strip()
    full_name = f"{first_name} {last_name}".strip()

    # Căutăm în CSV după nume complet sau last name
    match = df[df["name"].str.lower() == full_name.lower()]
    
    if match.empty:
        # Încercăm după last name
        match = df[df["last_name"].str.lower() == last_name.lower()]
    
    if match.empty:
        not_found += 1
        continue

    row = match.iloc[0]

    # Adăugăm datele în Firebase cu merge=True
    update_data = {}
    
    if pd.notna(row.get("market_value_in_eur")):
        update_data["marketValueEur"] = int(row["market_value_in_eur"])
    if pd.notna(row.get("highest_market_value_in_eur")):
        update_data["highestMarketValueEur"] = int(row["highest_market_value_in_eur"])
    if pd.notna(row.get("current_club_name")):
        update_data["tmClub"] = row["current_club_name"]
    if pd.notna(row.get("foot")):
        update_data["foot"] = row["foot"]
    if pd.notna(row.get("height_in_cm")):
        update_data["height"] = int(row["height_in_cm"])
    if pd.notna(row.get("contract_expiration_date")):
        update_data["contractExpiry"] = row["contract_expiration_date"]
    if pd.notna(row.get("position")):
        update_data["tmPosition"] = row["position"]
    if pd.notna(row.get("sub_position")):
        update_data["tmSubPosition"] = row["sub_position"]

    if update_data:
        db.collection("players").document(str(player["wyId"])).set(
            update_data, merge=True
        )
        print(f"  ✅ {full_name} → €{update_data.get('marketValueEur', 'N/A')} | {update_data.get('tmClub', 'N/A')}")
        enriched += 1
    else:
        not_found += 1

print(f"\n🎉 Gata! {enriched} enriched, {not_found} not found")