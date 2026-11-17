# debug_player.py
import firebase_admin
from firebase_admin import firestore

# Uses Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS locally).
firebase_admin.initialize_app()
db = firestore.client()

# Vedem primii 5 jucători din playerStats ca să vedem ce playerId-uri există
print("📋 Sample playerId-uri din playerStats:")
docs = db.collection("playerStats").limit(10).stream()
for doc in docs:
    data = doc.to_dict()
    print(f"  playerId: {data.get('playerId')} | matchLabel: {data.get('matchLabel', '')[:40]}")

# Vedem primii 5 jucători din players
print("\n👤 Sample wyId-uri din players:")
docs = db.collection("players").limit(10).stream()
for doc in docs:
    data = doc.to_dict()
    print(f"  wyId: {data.get('wyId')} | nume: {data.get('shortName')}")