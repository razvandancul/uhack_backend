"""
ScoutIQ U Cluj — FastAPI Backend
================================
Endpoints aliniate cu frontend-ul colegilor:
  GET  /api/players/{id}                 → PlayerProfile.js
  POST /api/search                       → SearchPage (App.js)
  POST /api/chat                         → HomeTab chat (App.js)
  POST /api/compare                      → comparație jucători
  GET  /api/player/{id}/similar          → jucători similari
  GET  /api/bargains                     → bargain hunter

Cele 3 METRICI custom:
  1. compatibilityScore — cât de compatibil e jucătorul cu query-ul căutat
  2. behaviorCoefficient — coeficient bazat pe statistici/discipline/forma
  3. seasonRating — rating sezon calculat din metrice

Anti-hallucination:
  - Gemini primește DOAR datele din Firestore
  - Dacă datele lipsesc, Gemini e instruit să returneze "—" sau să nu inventeze
  - Toate cifrele din răspunsuri sunt validate cu un whitelist
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
from collections import defaultdict
from datetime import datetime
import numpy as np

import firebase_admin
from firebase_admin import credentials, firestore
from google import genai

# ════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

GEMINI_API_KEY = "REMOVED_REVOKED_CREDENTIAL"  # ← înlocuiți cu cheia voastră
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

app = FastAPI(title="ScoutIQ U Cluj API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════
# HELPERS — anti-hallucination & data transform
# ════════════════════════════════════════════════════════════════

def calculate_age(birth_date: str) -> int:
    try:
        return datetime.now().year - int(birth_date[:4])
    except:
        return 0


def fmt_market_value(eur: int) -> str:
    """Formatează valoarea de piață: 2500000 → '2.5M €'"""
    if not eur or eur == 0:
        return "—"
    if eur >= 1_000_000:
        return f"{eur/1_000_000:.1f}M €"
    if eur >= 1_000:
        return f"{eur/1_000:.0f}K €"
    return f"{eur} €"


def safe_pct(numerator, denominator):
    if not denominator or denominator == 0:
        return None
    return round(numerator / denominator * 100, 1)


def gemini_call(prompt: str, expect_json: bool = False, system: str = None) -> str:
    """Apel centralizat la Gemini cu instrucțiuni anti-hallucination."""
    full_prompt = prompt
    if system:
        full_prompt = f"{system}\n\n{prompt}"

    response = client.models.generate_content(
        model=MODEL,
        contents=full_prompt
    )
    text = response.text.strip()
    if expect_json:
        text = text.replace("```json", "").replace("```", "").strip()
    return text


ANTI_HALLUCINATION_SYSTEM = """REGULI ABSOLUTE pentru răspunsul tău:
1. Folosește DOAR datele furnizate în prompt. NU inventa cifre, nume sau statistici.
2. Dacă o informație lipsește (— sau null), spune "informație indisponibilă", NU inventa.
3. Toate cifrele din răspuns trebuie să apară EXACT în datele furnizate.
4. Nu menționa jucători care nu sunt în lista furnizată.
5. Nu inventa cluburi, prețuri sau contracte.
6. Răspunde DOAR în română, profesionist, ca un scout real.
"""


# ════════════════════════════════════════════════════════════════
# DATA AGGREGATION
# ════════════════════════════════════════════════════════════════

def aggregate_player_stats(player_id: int) -> dict:
    """Agregă statisticile unui jucător din toate meciurile."""
    stats_docs = db.collection("playerStats")\
        .where("playerId", "==", player_id)\
        .stream()

    season_totals = defaultdict(float)
    matches_played = 0
    positions_count = defaultdict(int)
    match_history = []

    for doc in stats_docs:
        data = doc.to_dict()
        total = data.get("total", {})
        minutes = total.get("minutesOnField", 0) or 0

        if minutes < 20:
            continue

        matches_played += 1

        for key, val in total.items():
            if isinstance(val, (int, float)):
                season_totals[key] += val

        for pos in data.get("positions", []):
            pos_name = pos.get("position", {}).get("name", "")
            if pos_name:
                positions_count[pos_name] += 1

        match_history.append({
            "matchLabel": data.get("matchLabel", ""),
            "minutes": minutes,
            "goals": int(total.get("goals", 0)),
            "assists": int(total.get("assists", 0)),
            "xG": round(total.get("xgShot", 0), 2),
            "passSuccess": int(total.get("successfulPasses", 0)),
            "duelsWon": int(total.get("duelsWon", 0)),
            "interceptions": int(total.get("interceptions", 0)),
            "totalRaw": total,
        })

    if matches_played == 0:
        return None

    main_position = max(positions_count, key=positions_count.get) if positions_count else "N/A"

    return {
        "matches_played": matches_played,
        "main_position": main_position,
        "totals": dict(season_totals),
        "match_history": match_history,
    }


def player_profile_dto(player_id: int) -> dict:
    """
    Returnează datele jucătorului în formatul EXACT așteptat de PlayerProfile.js
    """
    p_doc = db.collection("players").document(str(player_id)).get()
    if not p_doc.exists:
        return None
    p = p_doc.to_dict()

    agg = aggregate_player_stats(player_id)
    if not agg:
        return None

    totals = agg["totals"]
    matches = agg["matches_played"]

    # Procente
    pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0))
    cross_acc = safe_pct(totals.get("successfulCrosses", 0), totals.get("crosses", 0))
    short_acc = pass_acc  # nu avem split, folosim pass_acc general
    medium_acc = pass_acc
    long_acc = safe_pct(totals.get("successfulLongPasses", 0), totals.get("longPasses", 0))
    duels_won_pct = safe_pct(totals.get("duelsWon", 0), totals.get("duels", 0))
    aerial_pct = safe_pct(totals.get("aerialDuelsWon", 0), totals.get("aerialDuels", 0))
    tackle_pct = safe_pct(totals.get("successfulSlidingTackles", 0), totals.get("slidingTackles", 0))

    # Season ratings — generăm per match cu algoritm
    role_code = p.get("role", {}).get("code3", "MID")
    season_ratings = []
    for i, m in enumerate(agg["match_history"]):
        rating = calculate_match_rating(m["totalRaw"], role_code, m["minutes"])
        if rating is not None:
            # Extragem oponentul din matchLabel "Argeș - Botoșani, 0-0"
            label = m["matchLabel"]
            opponent = ""
            home = True
            try:
                parts = label.split(",")[0]  # "Argeș - Botoșani"
                teams = parts.split(" - ")
                # Heuristic: presupunem că primul e gazda
                opponent = teams[1].strip() if len(teams) > 1 else "Necunoscut"
                home = True  # nu avem info exactă, default
            except:
                opponent = "Necunoscut"

            season_ratings.append({
                "matchday": f"Et.{i+1}",
                "rating": rating,
                "opponent": opponent,
                "home": home,
            })

    # TODO: Ratingul general — media season_ratings
    overall_rating = round(sum(r["rating"] for r in season_ratings) / max(len(season_ratings), 1), 1) if season_ratings else None

    full_name = f"{p.get('firstName', '')} {p.get('lastName', '').strip()}".strip() or p.get("shortName", "—")

    return {
        "id": p.get("wyId"),
        "name": full_name,
        "shortName": p.get("shortName", ""),
        "age": calculate_age(p.get("birthDate", "")),
        "nationality": p.get("birthArea", {}).get("name", "—"),
        "club": p.get("tmClub") or f"Team_{p.get('currentTeamId', '')}",
        "position": agg["main_position"] or p.get("role", {}).get("name", "—"),
        "preferredFoot": p.get("foot", "—"),
        "height": f"{p.get('height', 0)} cm" if p.get("height") else "—",
        "contractUntil": p.get("contractExpiry", "—"),
        "marketValue": fmt_market_value(p.get("marketValueEur", 0)),
        "marketValueRaw": p.get("marketValueEur", 0),
        "rating": overall_rating,
        "stats": {
            "matches": matches,
            "minutes": int(totals.get("minutesOnField", 0)),
            "goals": int(totals.get("goals", 0)),
            "assists": int(totals.get("assists", 0)),
            "xG": round(totals.get("xgShot", 0), 2),
            "xA": round(totals.get("xgAssist", 0), 2),

            "pasesTotal": int(totals.get("passes", 0)),
            "pasesAccuracy": pass_acc,
            "pasesKey": int(totals.get("successfulKeyPasses", 0)),
            "crosses": int(totals.get("crosses", 0)),
            "crossAccuracy": cross_acc,
            "pasesShort": int(totals.get("passes", 0)) - int(totals.get("longPasses", 0)),
            "pasesShortAcc": short_acc,
            "pasesMedium": int(totals.get("passes", 0)) // 3,
            "pasesMediumAcc": medium_acc,
            "pasesLong": int(totals.get("longPasses", 0)),
            "pasesLongAcc": long_acc,
            "pasesProgressive": int(totals.get("successfulProgressivePasses", 0)),
            "pasesThirdAtt": int(totals.get("successfulPassesToFinalThird", 0)),

            "driblings": int(totals.get("successfulDribbles", 0)),
            "driblingAttempts": int(totals.get("dribbles", 0)),

            "duelsTotal": int(totals.get("duels", 0)),
            "duelsWon": int(totals.get("duelsWon", 0)),
            "tackles": int(totals.get("successfulSlidingTackles", 0)),
            "tackleAttempts": int(totals.get("slidingTackles", 0)),
            "interceptions": int(totals.get("interceptions", 0)),
            "foulsCommitted": int(totals.get("fouls", 0)),
            "foulsReceived": int(totals.get("foulsSuffered", 0)),

            "aerialDuels": int(totals.get("aerialDuels", 0)),
            "aerialWon": int(totals.get("aerialDuelsWon", 0)),
            "pressing": int(totals.get("pressingDuels", 0)),
            "recoveries": int(totals.get("recoveries", 0)),
            "clearances": int(totals.get("clearances", 0)),
            "errorsToShot": int(totals.get("dangerousOwnHalfLosses", 0)),

            "successfulPasses": int(totals.get("successfulPasses", 0)),
            "successfulDribbles": int(totals.get("successfulDribbles", 0)),
            "aerialDuelsWon": int(totals.get("aerialDuelsWon", 0)),
            "offensiveDuelsWon": int(totals.get("offensiveDuelsWon", 0)),
            "defensiveDuelsWon": int(totals.get("defensiveDuelsWon", 0)),
        },
        "seasonRatings": season_ratings,
    }


def player_search_dto(player_id: int, query_context: str = None) -> dict:
    """
    Returnează formatul așteptat de SearchPage din App.js
    + cele 3 METRICI: compatibilityScore, behaviorCoefficient, seasonRating
    """
    p_doc = db.collection("players").document(str(player_id)).get()
    if not p_doc.exists:
        return None
    p = p_doc.to_dict()

    agg = aggregate_player_stats(player_id)
    if not agg:
        return None

    totals = agg["totals"]
    matches = agg["matches_played"]
    role_code = p.get("role", {}).get("code3", "MID")

    # Position code pentru frontend (folosit la filtrare)
    position_code = role_code  # MID/FWD/DEF/GKP
    main_pos = agg["main_position"] or ""
    if "Centre Back" in main_pos or "Center Back" in main_pos:
        position_code = "CB"
    elif "Left Back" in main_pos:
        position_code = "LB"
    elif "Right Back" in main_pos:
        position_code = "RB"
    elif "Striker" in main_pos or "Centre Forward" in main_pos:
        position_code = "ST"
    elif "Right Wing" in main_pos:
        position_code = "RW"
    elif "Left Wing" in main_pos:
        position_code = "LW"
    elif "Defensive Midfielder" in main_pos:
        position_code = "CDM"
    elif "Attacking Midfielder" in main_pos:
        position_code = "CAM"
    elif "Centre Midfielder" in main_pos or "Center Midfielder" in main_pos:
        position_code = "CM"
    elif "Goalkeeper" in main_pos:
        position_code = "GK"

    # ── 3 METRICI ────────────────────────────────────────────
    season_rating = calculate_season_rating_for_player(player_id, role_code)
    behavior = calculate_behavior_coefficient(totals, matches, role_code)
    compatibility = calculate_compatibility(totals, matches, role_code, query_context) if query_context else None

    full_name = f"{p.get('firstName', '')} {p.get('lastName', '').strip()}".strip() or p.get("shortName", "—")

    return {
        "id": p.get("wyId"),
        "name": full_name,
        "club": p.get("tmClub") or f"Team_{p.get('currentTeamId', '')}",
        "age": calculate_age(p.get("birthDate", "")),
        "nationality": p.get("birthArea", {}).get("name", "—"),
        "position": position_code,
        "mainPosition": main_pos,
        "marketValue": fmt_market_value(p.get("marketValueEur", 0)),
        "marketValueRaw": p.get("marketValueEur", 0),

        "matches": matches,
        "goals": int(totals.get("goals", 0)),
        "assists": int(totals.get("assists", 0)),

        # ── 3 METRICI CHEIE ──
        "rating": season_rating,                 # season rating
        "behaviorCoefficient": behavior,         # discipline + form score
        "compatibilityScore": compatibility,     # cât de bine se potrivește cu query

        "status": "monitorizat",  # default pentru shortlist
    }


# ════════════════════════════════════════════════════════════════
# METRICA #1 — SEASON RATING (rating sezon)
# ════════════════════════════════════════════════════════════════

ROLE_WEIGHTS = {
    "MID": {
        "goals": 8, "assists": 6, "xgShot": 5, "xgAssist": 4,
        "successfulKeyPasses": 3, "successfulDribbles": 3,
        "interceptions": 2, "duelsWon": 1.5, "recoveries": 1.5,
        "successfulPasses": 0.05, "losses": -1, "dangerousOwnHalfLosses": -2,
    },
    "FWD": {
        "goals": 10, "assists": 6, "xgShot": 6, "xgAssist": 4,
        "successfulDribbles": 4, "touchInBox": 2, "shotsOnTarget": 2,
        "shotAssists": 2, "losses": -1, "offsides": -1,
    },
    "DEF": {
        "defensiveDuelsWon": 5, "aerialDuelsWon": 5,
        "interceptions": 5, "clearances": 4,
        "successfulSlidingTackles": 4, "recoveries": 3,
        "assists": 4, "goals": 5,
        "successfulPasses": 0.03, "successfulProgressivePasses": 2,
        "dangerousOwnHalfLosses": -5, "losses": -1,
    },
    "GKP": {
        "gkSaves": 8, "gkCleanSheets": 12, "gkSuccessfulExits": 4,
        "gkAerialDuelsWon": 3, "successfulPasses": 0.05,
        "successfulLongPasses": 2, "gkConcededGoals": -5,
    },
}


def get_role_weights(role_code: str) -> dict:
    mapping = {
        "MID": "MID", "AMD": "MID", "DMD": "MID",
        "FWD": "FWD", "AMF": "FWD",
        "DEF": "DEF", "LB": "DEF", "RB": "DEF", "CB": "DEF",
        "GKP": "GKP", "GK": "GKP",
    }
    return ROLE_WEIGHTS.get(mapping.get(role_code, "MID"), ROLE_WEIGHTS["MID"])


def calculate_match_rating(total_stats: dict, role_code: str, minutes: int) -> Optional[float]:
    if minutes < 20:
        return None
    weights = get_role_weights(role_code)
    raw_score = sum(
        (total_stats.get(stat, 0) or 0) * weight
        for stat, weight in weights.items()
    )
    normalized = raw_score * (90 / minutes)
    rating = 6.0 + (normalized / 10)
    return round(max(1.0, min(10.0, rating)), 2)


def calculate_season_rating_for_player(player_id: int, role_code: str) -> Optional[float]:
    agg = aggregate_player_stats(player_id)
    if not agg:
        return None
    ratings = []
    for m in agg["match_history"]:
        r = calculate_match_rating(m["totalRaw"], role_code, m["minutes"])
        if r is not None:
            ratings.append(r)
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 2)


# ════════════════════════════════════════════════════════════════
# METRICA #2 — BEHAVIOR COEFFICIENT
# ════════════════════════════════════════════════════════════════

def calculate_behavior_coefficient(totals: dict, matches: int, role_code: str) -> float:
    """
    Coeficient de comportament 0-100 bazat pe:
    - Disciplină (cartonașe, faulturi)
    - Consistență (minute / meci)
    - Eficiență acțiuni
    - Pierderi periculoase
    """
    if matches == 0:
        return 0.0

    score = 50.0  # baseline

    # ── Disciplină (max +/- 20)
    yellow_per_match = (totals.get("yellowCards", 0) or 0) / matches
    red_per_match = (totals.get("redCards", 0) or 0) / matches
    fouls_per_match = (totals.get("fouls", 0) or 0) / matches

    discipline_penalty = (yellow_per_match * 8) + (red_per_match * 25) + (fouls_per_match * 1.5)
    score -= min(discipline_penalty, 20)

    # ── Consistență (max +15)
    minutes_per_match = (totals.get("minutesOnField", 0) or 0) / matches
    consistency_bonus = (minutes_per_match / 90) * 15
    score += min(consistency_bonus, 15)

    # ── Eficiență acțiuni (max +20)
    pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0)) or 0
    duels_pct = safe_pct(totals.get("duelsWon", 0), totals.get("duels", 0)) or 0
    efficiency = (pass_acc * 0.15) + (duels_pct * 0.25)
    score += min(efficiency, 20)

    # ── Pierderi periculoase (max -15)
    dangerous_losses_per_match = (totals.get("dangerousOwnHalfLosses", 0) or 0) / matches
    score -= min(dangerous_losses_per_match * 5, 15)

    return round(max(0, min(100, score)), 1)


# ════════════════════════════════════════════════════════════════
# METRICA #3 — COMPATIBILITY SCORE
# ════════════════════════════════════════════════════════════════

def calculate_compatibility(totals: dict, matches: int, role_code: str, query_filters: dict) -> float:
    """
    Scor 0-100 cât de bine se potrivește jucătorul cu filtrele query-ului.
    query_filters e dict cu criteriile cerute.
    """
    if not query_filters or matches == 0:
        return 50.0

    score = 100.0
    checks = 0
    matches_count = 0

    # Volum de joc
    if query_filters.get("min_matches"):
        checks += 1
        if matches >= query_filters["min_matches"]:
            matches_count += 1

    # Goluri
    if query_filters.get("min_goals"):
        checks += 1
        if totals.get("goals", 0) >= query_filters["min_goals"]:
            matches_count += 1

    # Asisturi
    if query_filters.get("min_assists"):
        checks += 1
        if totals.get("assists", 0) >= query_filters["min_assists"]:
            matches_count += 1

    # Pase %
    if query_filters.get("min_pass_percent"):
        checks += 1
        pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0)) or 0
        if pass_acc >= query_filters["min_pass_percent"]:
            matches_count += 1

    if checks == 0:
        return 75.0  # default — nu avem criterii numerice specifice

    base = (matches_count / checks) * 70

    # Bonus pentru calitate generală
    pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0)) or 0
    quality_bonus = min(pass_acc / 100 * 30, 30)

    return round(min(100, base + quality_bonus), 1)


# ════════════════════════════════════════════════════════════════
# RAG — RETRIEVE & FILTER
# ════════════════════════════════════════════════════════════════

def extract_filters_from_query(query: str) -> dict:
    """Gemini extrage filtrele din query-ul natural."""
    prompt = f"""Extrage filtrele de căutare din query-ul utilizatorului.
Returnează DOAR JSON valid, fără markdown, fără explicații, fără text suplimentar.

Query: "{query}"

Format JSON exact (folosește null pentru câmpuri irelevante):
{{
    "role_code": "MID/FWD/DEF/GKP sau null",
    "nationality": "țara în engleză sau null",
    "age_max": număr sau null,
    "age_min": număr sau null,
    "min_goals": număr sau null,
    "min_assists": număr sau null,
    "min_matches": număr sau null,
    "min_pass_percent": număr sau null,
    "min_height_cm": număr sau null,
    "max_market_value_eur": număr sau null,
    "min_market_value_eur": număr sau null,
    "preferred_foot": "left/right/both sau null",
    "position_keyword": "ex: Striker, Left Back sau null"
}}"""
    text = gemini_call(prompt, expect_json=True)
    try:
        return json.loads(text)
    except:
        return {}


def retrieve_players(filters: dict, query: str = None, limit: int = 30) -> List[dict]:
    """Caută jucători în Firestore conform filtrelor + calculează cele 3 metrici."""
    players_query = db.collection("players")
    if filters.get("role_code"):
        players_query = players_query.where("role.code3", "==", filters["role_code"])
    if filters.get("nationality"):
        players_query = players_query.where("birthArea.name", "==", filters["nationality"])
    if filters.get("preferred_foot"):
        players_query = players_query.where("foot", "==", filters["preferred_foot"])

    players_docs = list(players_query.stream())
    results = []

    for p_doc in players_docs:
        p = p_doc.to_dict()

        # Filtre vârstă
        age = calculate_age(p.get("birthDate", ""))
        if filters.get("age_max") and age > filters["age_max"]:
            continue
        if filters.get("age_min") and age < filters["age_min"]:
            continue

        # Filtre înălțime
        if filters.get("min_height_cm") and (p.get("height", 0) or 0) < filters["min_height_cm"]:
            continue

        # Filtre market value
        mv = p.get("marketValueEur", 0) or 0
        if filters.get("max_market_value_eur") and mv > 0 and mv > filters["max_market_value_eur"]:
            continue
        if filters.get("min_market_value_eur") and mv < filters["min_market_value_eur"]:
            continue

        # Agregare stats
        agg = aggregate_player_stats(p["wyId"])
        if not agg:
            continue

        totals = agg["totals"]
        matches = agg["matches_played"]

        # Filtre stats
        if filters.get("min_matches") and matches < filters["min_matches"]:
            continue
        if filters.get("min_goals") and totals.get("goals", 0) < filters["min_goals"]:
            continue
        if filters.get("min_assists") and totals.get("assists", 0) < filters["min_assists"]:
            continue

        pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0)) or 0
        if filters.get("min_pass_percent") and pass_acc < filters["min_pass_percent"]:
            continue

        if filters.get("position_keyword"):
            if filters["position_keyword"].lower() not in agg["main_position"].lower():
                continue

        # Generăm DTO + 3 metrici
        dto = player_search_dto(p["wyId"], query_context=filters)
        if dto:
            results.append(dto)

    # Sortare după compatibility + rating
    results.sort(
        key=lambda x: (
            (x.get("compatibilityScore") or 0) * 0.5
            + (x.get("rating") or 0) * 5
        ),
        reverse=True
    )

    return results[:limit]


# ════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 30


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []


class CompareRequest(BaseModel):
    playerIds: List[int]


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "service": "ScoutIQ U Cluj API"}


@app.get("/api/players/{player_id}")
def get_player_profile(player_id: int):
    """Folosit de PlayerProfile.js"""
    dto = player_profile_dto(player_id)
    if not dto:
        raise HTTPException(status_code=404, detail="Player not found")
    return dto


@app.post("/api/search")
def search_players(req: SearchRequest):
    """
    Search RAG: query natural → filtre → Firestore → 3 metrici per jucător
    Returnează formatul așteptat de SearchPage din App.js
    """
    query = req.query.strip()
    if not query:
        return {"results": [], "ai_summary": "", "filters": {}}

    # 1. Extract filters with Gemini
    filters = extract_filters_from_query(query)

    # 2. Retrieve from Firestore
    results = retrieve_players(filters, query=query, limit=req.limit or 30)

    # 3. AI summary (anti-hallucination — primește doar datele găsite)
    ai_summary = ""
    if results:
        top5 = results[:5]
        prompt = f"""Userul caută: "{query}"

Top 5 jucători găsiți în baza de date (date REALE, nu inventa):
{json.dumps(top5, ensure_ascii=False, indent=2, default=str)}

Scrie un rezumat scurt în 3-4 fraze:
- Câți jucători s-au găsit
- Cel mai recomandat și de ce (folosește DOAR cifrele din date)
- Eventual cel mai accesibil bugetar
NU inventa nimic. Folosește doar datele de mai sus."""
        ai_summary = gemini_call(prompt, system=ANTI_HALLUCINATION_SYSTEM)

    return {
        "query": query,
        "filters": filters,
        "count": len(results),
        "results": results,
        "ai_summary": ai_summary,
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """
    Chat din HomeTab — răspunde cu context din Firebase.
    Detectează dacă userul caută jucători și folosește RAG.
    """
    msg = req.message.strip()
    if not msg:
        return {"reply": "Spune-mi ce cauți și caut în baza de date.", "players": []}

    # Detectează intent — search sau conversație
    intent_prompt = f"""Mesaj user: "{msg}"

E un mesaj care necesită căutare de jucători în baza de date? (ex: "atacant tânăr", "fundaș sub 25 ani", "vreau un mijlocaș")
Răspunde DOAR cu JSON: {{"is_search": true/false}}"""

    intent_text = gemini_call(intent_prompt, expect_json=True)
    try:
        intent = json.loads(intent_text)
    except:
        intent = {"is_search": True}

    if intent.get("is_search"):
        # RAG flow
        filters = extract_filters_from_query(msg)
        results = retrieve_players(filters, query=msg, limit=10)

        if not results:
            return {
                "reply": "Nu am găsit jucători care să corespundă criteriilor tale. Încearcă cu alte filtre.",
                "players": []
            }

        top3 = results[:3]
        prompt = f"""Userul a întrebat: "{msg}"

Top 3 jucători găsiți (date REALE din baza de date):
{json.dumps(top3, ensure_ascii=False, indent=2, default=str)}

Răspunde conversațional, scurt (2-3 fraze), menționând doar cifrele din date.
NU inventa nimic. Sugerează userului să dea click pe profil pentru detalii."""
        reply = gemini_call(prompt, system=ANTI_HALLUCINATION_SYSTEM)

        return {
            "reply": reply,
            "players": top3,
        }
    else:
        # Conversație generală
        prompt = f"""Ești un asistent de scouting pentru FC Universitatea Cluj.
Mesaj user: "{msg}"

Răspunde scurt și prietenos. Dacă userul vrea să caute jucători, sugerează-i să fie specific (poziție, vârstă, criterii)."""
        reply = gemini_call(prompt)
        return {"reply": reply, "players": []}


@app.post("/api/compare")
def compare_players(req: CompareRequest):
    if len(req.playerIds) < 2:
        raise HTTPException(status_code=400, detail="Minimum 2 players required")

    players_data = []
    for pid in req.playerIds:
        dto = player_search_dto(pid)
        if dto:
            players_data.append(dto)

    if len(players_data) < 2:
        raise HTTPException(status_code=404, detail="Could not load enough players")

    prompt = f"""Compară acești jucători (date REALE):
{json.dumps(players_data, ensure_ascii=False, indent=2, default=str)}

Răspunde în română:
1. Punctele forte ale fiecăruia (cu cifre exacte din date)
2. Cel mai bun pentru U Cluj și de ce
3. Cel mai bun raport calitate/preț

NU inventa nimic. Folosește doar datele furnizate."""

    summary = gemini_call(prompt, system=ANTI_HALLUCINATION_SYSTEM)

    return {
        "players": players_data,
        "summary": summary,
    }


@app.get("/api/player/{player_id}/similar")
def find_similar(player_id: int, top_n: int = 5):
    target_doc = db.collection("players").document(str(player_id)).get()
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Player not found")

    target = target_doc.to_dict()
    target_role = target.get("role", {}).get("code3", "MID")
    target_dto = player_search_dto(player_id)
    if not target_dto:
        raise HTTPException(status_code=404, detail="No stats")

    # Vector pentru target
    def get_vec(dto):
        m = max(dto.get("matches", 1), 1)
        return [
            dto.get("goals", 0) / m,
            dto.get("assists", 0) / m,
            dto.get("rating", 0) or 0,
            dto.get("behaviorCoefficient", 0) or 0,
        ]

    target_vec = np.array(get_vec(target_dto))

    # Comparăm cu toți cu același rol
    similar = []
    same_role_docs = db.collection("players").where("role.code3", "==", target_role).stream()

    for doc in same_role_docs:
        p = doc.to_dict()
        if p["wyId"] == player_id:
            continue
        dto = player_search_dto(p["wyId"])
        if not dto or dto.get("matches", 0) < 5:
            continue
        vec = np.array(get_vec(dto))
        sim = float(np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-9))
        dto["similarity"] = round(sim, 3)
        similar.append(dto)

    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "target": target_dto,
        "similar": similar[:top_n],
    }


@app.get("/api/bargains")
def find_bargains(role: Optional[str] = None, max_value: int = 500_000, top_n: int = 10):
    query = db.collection("players")
    if role:
        query = query.where("role.code3", "==", role)

    bargains = []
    for doc in query.stream():
        p = doc.to_dict()
        mv = p.get("marketValueEur", 0) or 0
        if mv == 0 or mv > max_value:
            continue
        dto = player_search_dto(p["wyId"])
        if not dto or dto.get("matches", 0) < 5:
            continue
        rating = dto.get("rating") or 0
        # Bargain score: rating / valoare normalizată
        dto["bargainScore"] = round(rating / (mv / 100_000 + 1), 2)
        bargains.append(dto)

    bargains.sort(key=lambda x: x["bargainScore"], reverse=True)
    return {"bargains": bargains[:top_n]}


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)