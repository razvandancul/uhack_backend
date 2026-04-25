import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
from collections import defaultdict
import json

# ── Setup Firebase ─────────────────────────────────────────
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Setup Gemini ───────────────────────────────────────────
client = genai.Client(api_key="REMOVED_REVOKED_CREDENTIAL")

# ── Preia date jucător din Firestore ───────────────────────
def get_player_data(player_id: int):
    player_doc = db.collection("players").document(str(player_id)).get()
    if not player_doc.exists:
        return None
    player = player_doc.to_dict()

    stats_docs = db.collection("playerStats")\
        .where("playerId", "==", player_id)\
        .stream()

    season_totals = defaultdict(float)
    matches_played = 0

    for doc in stats_docs:
        total = doc.to_dict().get("total", {})
        if (total.get("minutesOnField") or 0) >= 20:
            matches_played += 1
            for k, v in total.items():
                if isinstance(v, (int, float)):
                    season_totals[k] += v

    return {
        "nume": player.get("shortName"),
        "rol": player.get("role", {}).get("name"),
        "nationalitate": player.get("birthArea", {}).get("name"),
        "meciuri": matches_played,
        "goluri": int(season_totals.get("goals", 0)),
        "asisturi": int(season_totals.get("assists", 0)),
        "xG": round(season_totals.get("xgShot", 0), 2),
        "pase_reușite_%": round(
            season_totals.get("successfulPasses", 0) /
            max(season_totals.get("passes", 1), 1) * 100, 1
        ),
        "dueluri_câștigate_%": round(
            season_totals.get("duelsWon", 0) /
            max(season_totals.get("duels", 1), 1) * 100, 1
        ),
    }

# ── Test ───────────────────────────────────────────────────
print("📡 Preluăm date din Firestore...")
data = get_player_data(1018334)
print(f"✅ Date: {data}")

print("🤖 Trimitem la Gemini...")
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=f"Ești un scout de fotbal. Analizează acest jucător și dă-i un rating 1-10 cu justificare: {json.dumps(data, ensure_ascii=False)}"
)

print("\n📋 RĂSPUNS GEMINI:")
print(response.text)