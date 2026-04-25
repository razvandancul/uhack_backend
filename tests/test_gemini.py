import json
import os
import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel

# ── 1. Setup Firebase ──────────────────────────────────────────
# Folosim cheia existentă
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ── 2. Setup Vertex AI ─────────────────────────────────────────
# IMPORTANT: Asigură-te că variabila de mediu punctează către cheia ta!
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "serviceAccountKey.json"

PROJECT_ID = "scoutingapp-494318"
LOCATION = "us-central1" # Rămânem pe us-central1 pentru că acolo sunt toate modelele active

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Folosim versiunea flash-002 care e stabilă pe Vertex
model = GenerativeModel("gemini-1.5-flash") # Fără -002

# ── Helper: calcul vârstă ─────────────────────────────────────
def calculate_age(birth_date: str) -> int:
    try:
        birth_year = int(birth_date[:4])
        return datetime.now().year - birth_year
    except:
        return 0

# ── 3. Funcția de Căutare (Query) ──────────────────────────────
def search_players_ai(user_query: str):
    print(f"\n🔍 Query: '{user_query}'")
    
    # Solicităm AI-ului să extragă filtrele în format JSON
    prompt_extract = f"""
    Ești un asistent de scouting. Extrage filtrele din acest query: "{user_query}"
    Returnează DOAR un obiect JSON valid, fără alte texte sau markdown.
    Format: {{"role_code": "MID/FWD/DEF/GKP", "age_max": numar, "nationality": "nume tara in engleza"}}
    """
    
    try:
        response = model.generate_content(prompt_extract)
        # Curățăm răspunsul de eventuale backticks (```json ... ```)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        filters = json.loads(clean_text)
        print(f"✅ Filtre identificate: {filters}")
    except Exception as e:
        print(f"❌ Eroare la procesarea AI: {e}")
        return

    # Interogare Firestore (Filtrare de bază pe Rol)
    players_query = db.collection("players")
    if filters.get("role_code"):
        players_query = players_query.where("role.code3", "==", filters["role_code"])
    
    docs = list(players_query.stream())
    results = []

    for doc in docs:
        p = doc.to_dict()
        age = calculate_age(p.get("birthDate", ""))
        
        # Filtrare manuală pe restul criteriilor (vârstă, naționalitate)
        if filters.get("age_max") and age > filters["age_max"]: continue
        if filters.get("nationality") and p.get("birthArea", {}).get("name") != filters["nationality"]: continue

        results.append({
            "nume": p.get("shortName"),
            "varsta": age,
            "nationalitate": p.get("birthArea", {}).get("name"),
            "rol": p.get("role", {}).get("name")
        })

    if not results:
        print("❌ Nu am găsit jucători care să corespundă.")
        return

    # Ranking final făcut de AI
    rank_prompt = f"""
    Analizează acești jucători și spune-mi cine se potrivește cel mai bine pentru cererea: "{user_query}"
    Jucători găsiți: {json.dumps(results, ensure_ascii=False)}
    Răspunde profesional în română.
    """
    
    final_response = model.generate_content(rank_prompt)
    print("\n🏆 RECOMANDĂRILE GEMINI (VERTEX AI):")
    print("-" * 40)
    print(final_response.text)

if __name__ == "__main__":
    # Testează query-ul tău
    search_players_ai("mijlocaș român sub 24 de ani")