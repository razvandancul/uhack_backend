# rag_service.py
from google import genai
from firebase_admin import firestore
import json
from collections import defaultdict

import firebase_admin
from firebase_admin import credentials
    
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()  

# ── Setup Gemini ──────────────────────────────────────────────
client = genai.Client(api_key="AIzaSyBzV-kWPffJ7-DGiqSAJdhY4n70u1JCicQ")  # cheia voastră

class ScoutRAG:
    """
    RAG = Retrieve → Augment → Generate
    
    Retrieve  = caută jucători în Firestore bazat pe query
    Augment   = adaugă datele găsite ca context pentru Gemini
    Generate  = Gemini generează răspunsul final
    """
    
    def __init__(self):
        self.model = "gemini-2.5-flash"
    
    # ── PASUL 1: RETRIEVE ─────────────────────────────────────
    def retrieve(self, query: str) -> list:
        """
        Primește query în română de la user.
        Gemini extrage filtrele → căutăm în Firestore → returnăm lista de jucători
        """
        
        # Gemini înțelege query-ul și îl transformă în filtre structurate
        filter_prompt = f"""
        Extrage filtrele din query pentru căutare jucători fotbal.
        Returnează DOAR JSON valid fără markdown.
        
        Query: "{query}"
        
        Format:
        {{
            "role_code": "MID/FWD/DEF/GKP sau null",
            "nationality": "tara in engleza sau null",
            "age_max": numar sau null,
            "age_min": numar sau null,
            "min_goals": numar sau null,
            "min_assists": numar sau null,
            "min_matches": numar sau null
        }}
        """
        
        response = client.models.generate_content(
            model=self.model,
            contents=filter_prompt
        )
        
        # Curățăm răspunsul și parsăm JSON-ul
        text = response.text.strip().replace("```json", "").replace("```", "")
        filters = json.loads(text)
        print(f"🔍 Filtre extrase de Gemini: {filters}")
        
        # Facem query în Firestore cu filtrele extrase
        players_query = db.collection("players")
        if filters.get("role_code"):
            players_query = players_query.where("role.code3", "==", filters["role_code"])
        if filters.get("nationality"):
            players_query = players_query.where("birthArea.name", "==", filters["nationality"])
        
        players = list(players_query.stream())
        print(f"📋 {len(players)} jucători găsiți în Firestore")
        
        # Filtrare suplimentară după vârstă și statistici
        # (Firestore nu suportă filtre complexe, le facem în Python)
        from datetime import datetime
        current_year = datetime.now().year
        results = []
        
        for p_doc in players:
            p = p_doc.to_dict()
            
            # Calculăm vârsta din birthDate
            birth = p.get("birthDate", "")
            age = current_year - int(birth[:4]) if birth else 0
            if filters.get("age_max") and age > filters["age_max"]:
                continue
            if filters.get("age_min") and age < filters["age_min"]:
                continue
            
            # Agregăm statisticile din toate meciurile sezonului
            stats = db.collection("playerStats")\
                .where("playerId", "==", p["wyId"])\
                .stream()
            
            totals = defaultdict(float)
            matches = 0
            positions = defaultdict(int)
            
            for s in stats:
                t = s.to_dict().get("total", {})
                # Ignorăm meciurile cu sub 20 minute
                if (t.get("minutesOnField") or 0) >= 20:
                    matches += 1
                    for k, v in t.items():
                        if isinstance(v, (int, float)):
                            totals[k] += v
                # Agregăm pozițiile jucate
                for pos in s.to_dict().get("positions", []):
                    pos_name = pos.get("position", {}).get("name", "")
                    if pos_name:
                        positions[pos_name] += 1
            
            # Filtre pe statistici
            if matches < (filters.get("min_matches") or 3):
                continue
            if filters.get("min_goals") and totals.get("goals", 0) < filters["min_goals"]:
                continue
            if filters.get("min_assists") and totals.get("assists", 0) < filters["min_assists"]:
                continue
            
            # Poziția predominantă din tot sezonul
            main_pos = max(positions, key=positions.get) if positions else "N/A"
            
            # Adăugăm jucătorul în rezultate cu statisticile agregate
            results.append({
                "wyId": p.get("wyId"),
                "nume": p.get("shortName"),
                "varsta": age,
                "nationalitate": p.get("birthArea", {}).get("name"),
                "rol": p.get("role", {}).get("name"),
                "pozitie": main_pos,
                "meciuri": matches,
                "goluri": int(totals.get("goals", 0)),
                "asisturi": int(totals.get("assists", 0)),
                "xG": round(totals.get("xgShot", 0), 2),
                "xA": round(totals.get("xgAssist", 0), 2),
                "pase_%": round(totals.get("successfulPasses", 0) / max(totals.get("passes", 1), 1) * 100, 1),
                "dueluri_%": round(totals.get("duelsWon", 0) / max(totals.get("duels", 1), 1) * 100, 1),
                "intercepții": int(totals.get("interceptions", 0)),
                "minute": int(totals.get("minutesOnField", 0)),
            })
        
        return results
    
    # ── PASUL 2: GENERATE ─────────────────────────────────────
    def generate(self, query: str, context: list) -> str:
        """
        Primește query original + lista de jucători găsiți.
        Gemini rankează și explică rezultatele în română.
        """
        if not context:
            return "Nu am găsit jucători care să corespundă criteriilor tale."
        
        generate_prompt = f"""
        Ești un scout profesionist la FC Universitatea Cluj.
        
        Userul caută: "{query}"
        
        Jucători găsiți în baza de date:
        {json.dumps(context, ensure_ascii=False, indent=2)}
        
        Răspunde în română cu:
        1. Top 5 jucători recomandați în ordine, cu justificare bazată pe statistici
        2. De ce fiecare e potrivit pentru query
        3. Unul pe care îl recomanzi cel mai tare și de ce
        
        Fii concis și profesionist.
        """
        
        response = client.models.generate_content(
            model=self.model,
            contents=generate_prompt
        )
        return response.text
    
    # ── PIPELINE COMPLET ──────────────────────────────────────
    def query(self, user_query: str) -> dict:
        """
        Funcția principală — asta o apelează FastAPI.
        Returnează un dict cu jucătorii găsiți + răspunsul AI.
        """
        print(f"\n{'='*50}")
        print(f"🚀 RAG Query: '{user_query}'")
        
        context = self.retrieve(user_query)   # Retrieve din Firestore
        answer = self.generate(user_query, context)  # Generate cu Gemini
        
        return {
            "query": user_query,
            "players_found": len(context),
            "players": context,      # ← React afișează cardurile din asta
            "ai_answer": answer      # ← React afișează textul AI din asta
        }

    # ── SCOUT REPORT ─────────────────────────────────────────
    def scout_report(self, player_id: int) -> dict:
        """
        Generează un raport complet pentru un jucător specific.
        Apelat de FastAPI la GET /api/player/{id}/report
        """
        # Luăm datele din Firestore
        player_doc = db.collection("players").document(str(player_id)).get()
        if not player_doc.exists:
            return {"error": "Jucătorul nu există"}
        
        player = player_doc.to_dict()
        stats = db.collection("playerStats")\
            .where("playerId", "==", player_id)\
            .stream()
        
        totals = defaultdict(float)
        matches = 0
        match_details = []
        positions = defaultdict(int)
        
        for s in stats:
            data = s.to_dict()
            t = data.get("total", {})
            minutes = t.get("minutesOnField", 0) or 0
            if minutes >= 20:
                matches += 1
                for k, v in t.items():
                    if isinstance(v, (int, float)):
                        totals[k] += v
                match_details.append({
                    "meci": data.get("matchLabel", ""),
                    "minute": minutes,
                    "goluri": t.get("goals", 0),
                    "xG": round(t.get("xgShot", 0), 2),
                })
            for pos in data.get("positions", []):
                pos_name = pos.get("position", {}).get("name", "")
                if pos_name:
                    positions[pos_name] += 1
        
        from datetime import datetime
        birth = player.get("birthDate", "")
        age = datetime.now().year - int(birth[:4]) if birth else 0
        main_pos = max(positions, key=positions.get) if positions else "N/A"
        
        player_data = {
            "info": {
                "nume": player.get("shortName"),
                "varsta": age,
                "rol": player.get("role", {}).get("name"),
                "nationalitate": player.get("birthArea", {}).get("name"),
                "pozitie_predominanta": main_pos,
            },
            "sezon": {
                "meciuri": matches,
                "goluri": int(totals.get("goals", 0)),
                "asisturi": int(totals.get("assists", 0)),
                "xG": round(totals.get("xgShot", 0), 2),
                "pase_%": round(totals.get("successfulPasses", 0) / max(totals.get("passes", 1), 1) * 100, 1),
                "dueluri_%": round(totals.get("duelsWon", 0) / max(totals.get("duels", 1), 1) * 100, 1),
                "intercepții": int(totals.get("interceptions", 0)),
            },
            "ultimele_5_meciuri": match_details[-5:]
        }
        
        # Gemini generează raportul
        prompt = f"""
        Ești un scout profesionist la FC Universitatea Cluj.
        Generează un raport de scouting profesionist în română pentru:
        
        {json.dumps(player_data, ensure_ascii=False, indent=2)}
        
        Include: profil general, calități, puncte slabe, 
        formă recentă, potrivire cu U Cluj, rating 1-10.
        """
        
        response = client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return {
            "player": player_data,
            "report": response.text
        }


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":    
    rag = ScoutRAG()
    
    # Test search
    result = rag.query("mijlocaș român sub 23 ani cu cel puțin 3 goluri")
    print("\n🤖 RĂSPUNS AI:")
    print(result["ai_answer"])
    
    # Test scout report
    report = rag.scout_report(1018334)
    print("\n📋 SCOUT REPORT:")
    
    print(report["report"])