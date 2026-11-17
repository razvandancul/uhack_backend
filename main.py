"""
ScoutIQ U Cluj — FastAPI Backend v3
Îmbunătățiri:
- Chat natural (CHAT_SYSTEM mai puțin restrictiv)
- Search după nume → 100% compatibility
- Search cu filtre din UI (/api/search/advanced)
- Scout report cu cache 1h
- Cache stats/players (5x mai rapid)
- Growth Potential score (0-100) în search + profile DTO
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import os
from collections import defaultdict
from datetime import datetime
import numpy as np

import firebase_admin
from firebase_admin import firestore
from google import genai

# Uses Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS locally).
firebase_admin.initialize_app()
db = firestore.client()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY before starting the backend.")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3-flash-preview"

app = FastAPI(title="ScoutIQ U Cluj API v3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── CACHE ────────────────────────────────────────────────────
_PLAYER_CACHE = {}
_STATS_CACHE = {}
_REPORT_CACHE = {}
_ALL_PLAYERS_CACHE = None


def get_player_doc(player_id: int):
    if player_id in _PLAYER_CACHE:
        return _PLAYER_CACHE[player_id]
    doc = db.collection("players").document(str(player_id)).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    _PLAYER_CACHE[player_id] = data
    return data


def get_all_players_cached():
    global _ALL_PLAYERS_CACHE
    if _ALL_PLAYERS_CACHE is None:
        docs = db.collection("players").stream()
        _ALL_PLAYERS_CACHE = [d.to_dict() for d in docs]
    return _ALL_PLAYERS_CACHE


# ── HELPERS ──────────────────────────────────────────────────
def calculate_age(birth_date):
    try: return datetime.now().year - int(birth_date[:4])
    except: return 0


def fmt_market_value(eur):
    if not eur or eur == 0: return "—"
    if eur >= 1_000_000: return f"{eur/1_000_000:.1f}M €"
    if eur >= 1_000: return f"{eur/1_000:.0f}K €"
    return f"{eur} €"


def safe_pct(num, den):
    if not den or den == 0: return None
    return round(num / den * 100, 1)


def normalize_name(name):
    if not name: return ""
    repl = {"ă":"a","â":"a","î":"i","ș":"s","ț":"t","Ă":"a","Â":"a","Î":"i","Ș":"s","Ț":"t"}
    s = name.lower().strip()
    for k, v in repl.items(): s = s.replace(k, v)
    return s


def gemini_call(prompt, expect_json=False, system=None):
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = client.models.generate_content(model=MODEL, contents=full)
    text = resp.text.strip()
    if expect_json:
        text = text.replace("```json", "").replace("```", "").strip()
    return text


# ── SYSTEM PROMPTS ───────────────────────────────────────────
CHAT_SYSTEM = """Ești un scout AI pentru FC Universitatea Cluj. Răspunzi STRICT pe baza datelor furnizate în prompt.

REGULI ABSOLUTE — nu le încălca niciodată:
1. Vorbești DOAR despre jucătorii menționați explicit în datele din prompt.
2. Dacă userul întreabă despre un jucător specific, discuți NUMAI despre acel jucător — nu sugera alții dacă nu ți s-a cerut.
3. NU inventa jucători, cluburi, statistici sau transferuri care nu apar în date.
4. NU spune niciodată că "există X jucători care ar putea completa lotul" dacă nu ai o listă concretă.
5. Dacă datele nu conțin informația cerută, spune: "Nu am date suficiente despre acest subiect."
6. Citează NUMAI cifrele exacte din datele furnizate.
7. Răspunsuri 2-3 fraze, concise și la obiect.
8. Limba română, ton de scout profesionist."""

SUMMARY_SYSTEM = """Scrie rezumate scurte de scouting după căutare în baza de date.
Reguli: cifre exacte din lista furnizată, 2-3 fraze, română, ton profesionist.
NU inventa jucători sau statistici care nu sunt în listă."""


# ── AGGREGATION ──────────────────────────────────────────────
def aggregate_player_stats(player_id):
    if player_id in _STATS_CACHE:
        return _STATS_CACHE[player_id]

    stats_docs = db.collection("playerStats").where("playerId", "==", player_id).stream()

    season_totals = defaultdict(float)
    matches_played = 0
    positions_count = defaultdict(int)
    match_history = []

    for doc in stats_docs:
        data = doc.to_dict()
        total = data.get("total", {})
        minutes = total.get("minutesOnField", 0) or 0
        if minutes < 15: continue

        matches_played += 1
        for k, v in total.items():
            if isinstance(v, (int, float)):
                season_totals[k] += v

        for pos in data.get("positions", []):
            pn = pos.get("position", {}).get("name", "")
            if pn: positions_count[pn] += 1

        match_history.append({
            "matchLabel": data.get("matchLabel", ""),
            "minutes": minutes,
            "goals": int(total.get("goals", 0)),
            "assists": int(total.get("assists", 0)),
            "xG": round(total.get("xgShot", 0), 2),
            "totalRaw": total,
        })

    if matches_played == 0:
        result = None
    else:
        main_pos = max(positions_count, key=positions_count.get) if positions_count else "N/A"
        result = {
            "matches_played": matches_played,
            "main_position": main_pos,
            "totals": dict(season_totals),
            "match_history": match_history,
        }
    _STATS_CACHE[player_id] = result
    return result


# ── RATING (Opta-style) ──────────────────────────────────────
ROLE_WEIGHTS = {
    "MID": {"goals":1.2,"assists":0.8,"successfulKeyPasses":0.15,"successfulPasses":0.02,
            "successfulDribbles":0.1,"xgShot":0.1,"xgAssist":0.08,"interceptions":0.05,
            "successfulSlidingTackles":0.1,"recoveries":0.04,"yellowCards":-0.3,
            "dangerousOwnHalfLosses":-0.25,"losses":-0.04},
    "FWD": {"goals":1.2,"assists":0.8,"successfulDribbles":0.1,"touchInBox":0.05,
            "xgShot":0.15,"xgAssist":0.08,"successfulKeyPasses":0.1,"shotsOnTarget":0.1,
            "successfulPasses":0.015,"yellowCards":-0.3,"offsides":-0.1,"losses":-0.04,
            "dangerousOwnHalfLosses":-0.2},
    "DEF": {"defensiveDuelsWon":0.12,"aerialDuelsWon":0.12,"interceptions":0.12,
            "clearances":0.08,"successfulSlidingTackles":0.1,"recoveries":0.06,
            "successfulProgressivePasses":0.05,"successfulPasses":0.015,"goals":0.8,
            "assists":0.6,"dangerousOwnHalfLosses":-0.5,"yellowCards":-0.3,"losses":-0.05},
    "GKP": {"gkSaves":0.3,"gkCleanSheets":0.8,"gkSuccessfulExits":0.1,"gkAerialDuelsWon":0.08,
            "successfulLongPasses":0.04,"successfulPasses":0.01,"gkConcededGoals":-0.4,
            "yellowCards":-0.3},
}
ROLE_MAP = {"MID":"MID","AMD":"MID","DMD":"MID","FWD":"FWD","AMF":"FWD",
            "DEF":"DEF","LB":"DEF","RB":"DEF","CB":"DEF","GKP":"GKP","GK":"GKP"}


def get_role_weights(rc):
    return ROLE_WEIGHTS.get(ROLE_MAP.get(rc, "MID"), ROLE_WEIGHTS["MID"])


def calculate_match_rating(total, role_code, minutes):
    has_impact = total.get("goals",0)>0 or total.get("assists",0)>0 or total.get("redCards",0)>0
    if minutes < 15 and not has_impact: return None

    rating = 6.0
    weights = get_role_weights(role_code)
    rating += sum((total.get(s, 0) or 0) * w for s, w in weights.items())

    if minutes >= 60 and total.get("gkConcededGoals", 1) == 0:
        if role_code in ("GKP","GK","DEF","CB","LB","RB"): rating += 0.5

    if total.get("redCards", 0) > 0: rating = min(rating, 5.5)
    errors = total.get("dangerousOwnHalfLosses", 0) or 0
    if errors > 0: rating -= 0.8 * errors

    return round(max(1.0, min(10.0, rating)), 2)


def calculate_season_rating_for_player(player_id, role_code):
    agg = aggregate_player_stats(player_id)
    if not agg: return None
    ratings = [calculate_match_rating(m["totalRaw"], role_code, m["minutes"])
               for m in agg["match_history"]]
    ratings = [r for r in ratings if r is not None]
    if not ratings: return None
    return round(sum(ratings) / len(ratings), 2)


def calculate_behavior_coefficient(totals, matches, role_code):
    if matches == 0: return 0.0
    score = 50.0
    yp = (totals.get("yellowCards",0) or 0) / matches
    rp = (totals.get("redCards",0) or 0) / matches
    fp = (totals.get("fouls",0) or 0) / matches
    score -= min((yp*8) + (rp*25) + (fp*1.5), 20)

    mp = (totals.get("minutesOnField",0) or 0) / matches
    score += min((mp/90)*15, 15)

    pa = safe_pct(totals.get("successfulPasses",0), totals.get("passes",0)) or 0
    dp = safe_pct(totals.get("duelsWon",0), totals.get("duels",0)) or 0
    score += min((pa*0.15) + (dp*0.25), 20)

    dl = (totals.get("dangerousOwnHalfLosses",0) or 0) / matches
    score -= min(dl*5, 15)

    return round(max(0.0, min(100.0, score)), 1)


def calculate_compatibility(totals, matches, role_code, query_filters, exact_name_match=False):
    if exact_name_match: return 100.0
    if not query_filters or matches == 0: return 50.0

    checks, hits = 0, 0
    criteria = [
        ("min_matches", lambda: matches >= query_filters["min_matches"]),
        ("min_goals", lambda: totals.get("goals", 0) >= query_filters["min_goals"]),
        ("min_assists", lambda: totals.get("assists", 0) >= query_filters["min_assists"]),
        ("min_pass_percent", lambda: (safe_pct(totals.get("successfulPasses",0), totals.get("passes",0)) or 0) >= query_filters["min_pass_percent"]),
    ]
    for k, fn in criteria:
        if query_filters.get(k):
            checks += 1
            if fn(): hits += 1
    if checks == 0: return 75.0

    base = (hits/checks) * 70
    pa = safe_pct(totals.get("successfulPasses",0), totals.get("passes",0)) or 0
    return round(min(100.0, base + min(pa/100*30, 30)), 1)


# ── GROWTH POTENTIAL ─────────────────────────────────────────
def calculate_growth_potential(age, season_rating, totals, matches, behavior_coefficient):
    """
    Scor 0–100 pentru TOȚI jucătorii, care reflectă potențialul de creștere
    sau de menținere a valorii. Apare în UI cu label contextual bazat pe vârstă.

    Componente:
      A. Rating curent         (40 pts) — performanța prezentă e baza
      B. Bonus vârstă          (35 pts) — tinerii au plafon mai mare; veteranii pot fi buni dar scor mai mic
      C. Comportament          (15 pts) — disciplina susține evoluția
      D. Consistență meciuri   (10 pts) — regularitatea în sezon

    Interpretare scor:
      ≥ 75  → "⚡ Top Prospect"   (tânăr + performanță ridicată)
      ≥ 55  → "🌱 Talent"         (în creștere, potențial clar)
      ≥ 35  → "📊 Consolidat"     (jucător stabil, valoare actuală)
      <  35 → "🔻 În declin"      (vârstă înaintată + performanță scăzută)
    """
    if matches == 0 or season_rating is None:
        return None

    score = 0.0

    # ── A. RATING CURENT (max 40 pts) ───────────────────────
    # Normalizăm rating (1–10) → (0–1) și scalăm.
    # Un rating de 7.0 dă ~33 pts; 8.5 dă ~39 pts; 5.0 dă ~22 pts.
    rating_norm = max(0.0, (season_rating - 1.0) / 9.0)
    score += rating_norm * 40.0

    # ── B. BONUS VÂRSTĂ (max 35 pts) ────────────────────────
    # Curba reflectă fereastra de creștere reală din fotbal:
    #   ≤ 19 ani → 35 pts (talent pur, tot plafon disponibil)
    #   20–21    → 32 pts
    #   22–23    → 27 pts (U23, target principal de scouting)
    #   24–26    → 18 pts (prime years, creștere limitată dar reală)
    #   27–29    → 8 pts  (maturitate, stabilitate)
    #   30–32    → 2 pts  (veteran, valoare decrescătoare)
    #   33+      → 0 pts
    if age <= 19:
        age_bonus = 35.0
    elif age <= 21:
        age_bonus = 35.0 - (age - 19) * 1.5      # 35 → 32
    elif age <= 23:
        age_bonus = 32.0 - (age - 21) * 2.5      # 32 → 27
    elif age <= 26:
        age_bonus = 27.0 - (age - 23) * 3.0      # 27 → 18
    elif age <= 29:
        age_bonus = 18.0 - (age - 26) * 3.33     # 18 → 8
    elif age <= 32:
        age_bonus = 8.0 - (age - 29) * 2.0       # 8 → 2
    else:
        age_bonus = max(0.0, 2.0 - (age - 32) * 1.0)

    score += age_bonus

    # ── C. COMPORTAMENT (max 15 pts) ────────────────────────
    score += (behavior_coefficient / 100.0) * 15.0

    # ── D. CONSISTENȚĂ (max 10 pts) ─────────────────────────
    # Jucătorii cu mai multe meciuri sunt mai de încredere.
    # Pragul de referință: 25 meciuri = sezon complet în Superliga.
    consistency = min(matches / 25.0, 1.0)
    score += consistency * 10.0

    return round(min(100.0, max(0.0, score)), 1)


# ── DTOs ─────────────────────────────────────────────────────
def player_search_dto(player_id, query_filters=None, exact_name_match=False):
    p = get_player_doc(player_id)
    if not p: return None
    agg = aggregate_player_stats(player_id)
    if not agg: return None

    totals = agg["totals"]
    matches = agg["matches_played"]
    role_code = p.get("role", {}).get("code3", "MID")

    position_code = role_code
    main_pos = agg["main_position"] or ""
    pos_map = {"Centre Back":"CB","Center Back":"CB","Left Back":"LB","Right Back":"RB",
               "Striker":"ST","Centre Forward":"ST","Right Wing":"RW","Left Wing":"LW",
               "Defensive Midfielder":"CDM","Attacking Midfielder":"CAM",
               "Centre Midfielder":"CM","Center Midfielder":"CM","Goalkeeper":"GK"}
    for k, c in pos_map.items():
        if k in main_pos: position_code = c; break

    full_name = f"{p.get('firstName','')} {p.get('lastName','').strip()}".strip() or p.get("shortName", "—")
    age = calculate_age(p.get("birthDate", ""))
    season_rating = calculate_season_rating_for_player(player_id, role_code)
    behavior = calculate_behavior_coefficient(totals, matches, role_code)
    growth = calculate_growth_potential(age, season_rating, totals, matches, behavior)

    return {
        "id": p.get("wyId"),
        "name": full_name,
        "club": p.get("tmClub") or f"Team_{p.get('currentTeamId','')}",
        "age": age,
        "nationality": p.get("birthArea", {}).get("name", "—"),
        "position": position_code,
        "mainPosition": main_pos,
        "marketValue": fmt_market_value(p.get("marketValueEur", 0)),
        "marketValueRaw": p.get("marketValueEur", 0),
        "matches": matches,
        "goals": int(totals.get("goals", 0)),
        "assists": int(totals.get("assists", 0)),
        "rating": season_rating,
        "behaviorCoefficient": behavior,
        "compatibilityScore": calculate_compatibility(totals, matches, role_code, query_filters, exact_name_match),
        "growthPotential": growth,
        "status": "monitorizat",
    }


def player_profile_dto(player_id):
    p = get_player_doc(player_id)
    if not p: return None
    agg = aggregate_player_stats(player_id)
    if not agg: return None

    totals = agg["totals"]
    matches = agg["matches_played"]

    pass_acc = safe_pct(totals.get("successfulPasses",0), totals.get("passes",0))
    cross_acc = safe_pct(totals.get("successfulCrosses",0), totals.get("crosses",0))
    long_acc = safe_pct(totals.get("successfulLongPasses",0), totals.get("longPasses",0))

    role_code = p.get("role", {}).get("code3", "MID")
    season_ratings = []
    for i, m in enumerate(agg["match_history"]):
        rating = calculate_match_rating(m["totalRaw"], role_code, m["minutes"])
        if rating is not None:
            try:
                parts = m["matchLabel"].split(",")[0]
                teams = parts.split(" - ")
                opponent = teams[1].strip() if len(teams) > 1 else "Necunoscut"
            except:
                opponent = "Necunoscut"
            season_ratings.append({
                "matchday": f"Et.{i+1}", "rating": rating,
                "opponent": opponent, "home": True,
            })

    overall = round(sum(r["rating"] for r in season_ratings) / max(len(season_ratings),1), 1) if season_ratings else None
    full_name = f"{p.get('firstName','')} {p.get('lastName','').strip()}".strip() or p.get("shortName", "—")
    age = calculate_age(p.get("birthDate", ""))
    behavior = calculate_behavior_coefficient(totals, matches, role_code)
    growth = calculate_growth_potential(age, overall, totals, matches, behavior)

    return {
        "id": p.get("wyId"),
        "name": full_name,
        "shortName": p.get("shortName",""),
        "age": age,
        "nationality": p.get("birthArea", {}).get("name", "—"),
        "club": p.get("tmClub") or f"Team_{p.get('currentTeamId','')}",
        "position": agg["main_position"] or p.get("role", {}).get("name", "—"),
        "preferredFoot": p.get("foot", "—"),
        "height": f"{p.get('height',0)} cm" if p.get("height") else "—",
        "contractUntil": p.get("contractExpiry", "—"),
        "marketValue": fmt_market_value(p.get("marketValueEur", 0)),
        "marketValueRaw": p.get("marketValueEur", 0),
        "rating": overall,
        "growthPotential": growth,
        "stats": {
            "matches": matches, "minutes": int(totals.get("minutesOnField",0)),
            "goals": int(totals.get("goals",0)), "assists": int(totals.get("assists",0)),
            "xG": round(totals.get("xgShot",0),2), "xA": round(totals.get("xgAssist",0),2),
            "pasesTotal": int(totals.get("passes",0)), "pasesAccuracy": pass_acc,
            "pasesKey": int(totals.get("successfulKeyPasses",0)),
            "crosses": int(totals.get("crosses",0)), "crossAccuracy": cross_acc,
            "pasesShort": int(totals.get("passes",0)) - int(totals.get("longPasses",0)),
            "pasesShortAcc": pass_acc,
            "pasesMedium": int(totals.get("passes",0)) // 3, "pasesMediumAcc": pass_acc,
            "pasesLong": int(totals.get("longPasses",0)), "pasesLongAcc": long_acc,
            "pasesProgressive": int(totals.get("successfulProgressivePasses",0)),
            "pasesThirdAtt": int(totals.get("successfulPassesToFinalThird",0)),
            "driblings": int(totals.get("successfulDribbles",0)),
            "driblingAttempts": int(totals.get("dribbles",0)),
            "duelsTotal": int(totals.get("duels",0)),
            "duelsWon": int(totals.get("duelsWon",0)),
            "tackles": int(totals.get("successfulSlidingTackles",0)),
            "tackleAttempts": int(totals.get("slidingTackles",0)),
            "interceptions": int(totals.get("interceptions",0)),
            "foulsCommitted": int(totals.get("fouls",0)),
            "foulsReceived": int(totals.get("foulsSuffered",0)),
            "aerialDuels": int(totals.get("aerialDuels",0)),
            "aerialWon": int(totals.get("aerialDuelsWon",0)),
            "pressing": int(totals.get("pressingDuels",0)),
            "recoveries": int(totals.get("recoveries",0)),
            "clearances": int(totals.get("clearances",0)),
            "errorsToShot": int(totals.get("dangerousOwnHalfLosses",0)),
            "successfulPasses": int(totals.get("successfulPasses",0)),
            "successfulDribbles": int(totals.get("successfulDribbles",0)),
            "aerialDuelsWon": int(totals.get("aerialDuelsWon",0)),
            "offensiveDuelsWon": int(totals.get("offensiveDuelsWon",0)),
            "defensiveDuelsWon": int(totals.get("defensiveDuelsWon",0)),
        },
        "seasonRatings": season_ratings,
    }


# ── NAME SEARCH ──────────────────────────────────────────────
def search_by_name(query, max_results=5):
    qn = normalize_name(query)
    if len(qn) < 2: return []

    all_players = get_all_players_cached()
    exact, partial = [], []

    for p in all_players:
        full = normalize_name(f"{p.get('firstName','')} {p.get('lastName','')}")
        short = normalize_name(p.get("shortName",""))
        last = normalize_name(p.get("lastName",""))

        if qn == full or qn == short or qn == last:
            exact.append((p, True))
        elif qn in full or qn in short or full.startswith(qn) or last.startswith(qn):
            partial.append((p, False))

    matches = exact + partial
    results = []
    for p, is_exact in matches[:max_results]:
        dto = player_search_dto(p["wyId"], query_filters=None, exact_name_match=is_exact)
        if dto: results.append(dto)
    return results


def is_name_query(query):
    q = query.strip().lower()
    exclude = ["atacant","fundas","fundaș","mijlocas","mijlocaș","portar","tanar","tânăr",
               "sub","peste","ani","goluri","asisturi","creativ","rapid","ieftin","buget",
               "cu ","fara","fără","min ","max ","centra","stang","drept","young","old",
               "spune","despre","jucatorul","jucătorul","nostru","echipa","echipă",
               "poti","poți","ceva","informatii","informații",
               "potential","potențial","prospect","talent","crestere","creștere",
               "viitor","promitator","promițător","growth"]
    for kw in exclude:
        if kw in q: return False
    if len(q.split()) > 4: return False
    if any(c.isdigit() for c in q): return False
    return True


# ── RAG ──────────────────────────────────────────────────────
def extract_filters_from_query(query):
    prompt = f"""Extrage filtrele pentru scouting fotbal. Returnează DOAR JSON valid.

Query: "{query}"

IMPORTANT pentru min_growth_potential:
- "potențial mare/ridicat/bun/top" → 70
- "potențial mediu" → 50
- "tineri cu potențial" → 65
- "cei mai promițători/prospect/talent" → 70
- "jucători în creștere" → 60
- dacă nu se menționează potențial → null

Format (null pentru irelevante):
{{"role_code": "MID/FWD/DEF/GKP sau null", "nationality": "țara în engleză sau null",
"age_max": număr sau null, "age_min": număr sau null,
"min_goals": număr sau null, "min_assists": număr sau null,
"min_matches": număr sau null, "min_pass_percent": număr sau null,
"min_height_cm": număr sau null, "max_market_value_eur": număr sau null,
"min_market_value_eur": număr sau null, "preferred_foot": "left/right/both sau null",
"position_keyword": "ex: Striker, Left Back sau null",
"min_growth_potential": număr 0-100 sau null,
"sort_by_growth": true dacă se cere explicit sortare după potențial, altfel false}}"""
    text = gemini_call(prompt, expect_json=True)
    try: return json.loads(text)
    except: return {}


def retrieve_players(filters, query=None, limit=30):
    q = db.collection("players")
    if filters.get("role_code"): q = q.where("role.code3", "==", filters["role_code"])
    if filters.get("nationality"): q = q.where("birthArea.name", "==", filters["nationality"])
    if filters.get("preferred_foot"): q = q.where("foot", "==", filters["preferred_foot"])

    docs = list(q.stream())
    results = []

    for d in docs:
        p = d.to_dict()
        _PLAYER_CACHE[p["wyId"]] = p

        age = calculate_age(p.get("birthDate", ""))
        if filters.get("age_max") and age > filters["age_max"]: continue
        if filters.get("age_min") and age < filters["age_min"]: continue
        if filters.get("min_height_cm") and (p.get("height",0) or 0) < filters["min_height_cm"]: continue

        mv = p.get("marketValueEur", 0) or 0
        if filters.get("max_market_value_eur") and mv > 0 and mv > filters["max_market_value_eur"]: continue
        if filters.get("min_market_value_eur") and mv < filters["min_market_value_eur"]: continue

        agg = aggregate_player_stats(p["wyId"])
        if not agg: continue

        totals = agg["totals"]
        matches = agg["matches_played"]

        if filters.get("min_matches") and matches < filters["min_matches"]: continue
        if filters.get("min_goals") and totals.get("goals",0) < filters["min_goals"]: continue
        if filters.get("min_assists") and totals.get("assists",0) < filters["min_assists"]: continue

        pa = safe_pct(totals.get("successfulPasses",0), totals.get("passes",0)) or 0
        if filters.get("min_pass_percent") and pa < filters["min_pass_percent"]: continue

        if filters.get("position_keyword"):
            if filters["position_keyword"].lower() not in agg["main_position"].lower(): continue

        dto = player_search_dto(p["wyId"], query_filters=filters)
        if not dto: continue

        # Filtrare după growth potential — calculat deja în DTO
        if filters.get("min_growth_potential"):
            gp = dto.get("growthPotential")
            if gp is None or gp < filters["min_growth_potential"]:
                continue

        results.append(dto)

    # Sortare: dacă s-a cerut explicit potențial → sortăm după growthPotential desc
    # Altfel → sortarea standard după compatibilitate + rating
    if filters.get("sort_by_growth") or filters.get("min_growth_potential"):
        results.sort(key=lambda x: (x.get("growthPotential") or 0), reverse=True)
    else:
        results.sort(
            key=lambda x: ((x.get("compatibilityScore") or 0) * 0.5 + (x.get("rating") or 0) * 5),
            reverse=True,
        )

    return results[:limit]


# ── SCOUT REPORT ─────────────────────────────────────────────
SCOUT_REPORT_SYSTEM = """Scout profesionist FC Universitatea Cluj.
Reguli: cifre exacte din date, fără invenții, fără comparații cu jucători celebri.
Returnează DOAR JSON valid."""


def build_report_prompt(profile, stats, metrics):
    return f"""Date REALE jucător:
Nume: {profile.get('name')} | Poziție: {profile.get('position')} | Vârstă: {profile.get('age')}
Club: {profile.get('club')} | Valoare: {profile.get('marketValue')}

Statistici sezon ({stats.get('matches',0)} meciuri):
- Goluri: {stats.get('goals',0)} | Asisturi: {stats.get('assists',0)} | xG: {stats.get('xG',0)} | xA: {stats.get('xA',0)}
- Pase: {stats.get('successfulPasses',0)}/{stats.get('pasesTotal',0)} ({stats.get('pasesAccuracy') or 0}%)
- Driblinguri: {stats.get('driblings',0)}/{stats.get('driblingAttempts',0)}
- Dueluri: {stats.get('duelsWon',0)}/{stats.get('duelsTotal',0)}
- Aeriene: {stats.get('aerialWon',0)}/{stats.get('aerialDuels',0)}
- Intercepții: {stats.get('interceptions',0)} | Recuperări: {stats.get('recoveries',0)}
- Galbene: {int(metrics.get('yellowCards',0))} | Roșii: {int(metrics.get('redCards',0))}

Metrici: Rating {metrics.get('seasonRating')}/10 | Comportament {metrics.get('behaviorCoefficient')}/100

Returnează DOAR JSON:
{{
  "verdict": {{
    "text": "3-5 fraze evaluare cu cifre",
    "recommendation": "Transfer recomandat/Transfer fezabil/Observare suplimentară/Nu se recomandă",
    "badges": ["max 3 badge-uri scurte"]
  }},
  "domains": {{
    "constructie": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "faza_ofensiva": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "faza_defensiva": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "dueluri_fizice": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "disciplina": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }}
  }},
  "strengths": ["3 puncte forte cu cifre"],
  "weaknesses": ["2 puncte slabe cu cifre"],
  "scout_recommendation": "2-3 fraze despre folosirea la U Cluj"
}}"""


def generate_scout_report(player_id):
    if player_id in _REPORT_CACHE:
        cached = _REPORT_CACHE[player_id]
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(cached["meta"]["generatedAt"].replace("Z",""))).total_seconds()
            if age < 3600: return cached
        except: pass

    profile = player_profile_dto(player_id)
    if not profile: return {"error": "Jucătorul nu a fost găsit."}

    stats = profile.get("stats", {})
    agg = aggregate_player_stats(player_id)
    if not agg: return {"error": "Date insuficiente."}

    p = get_player_doc(player_id)
    role_code = p.get("role", {}).get("code3", "MID") if p else "MID"

    behavior = calculate_behavior_coefficient(agg["totals"], agg["matches_played"], role_code)
    season_rating = calculate_season_rating_for_player(player_id, role_code)

    metrics = {
        "yellowCards": agg["totals"].get("yellowCards", 0),
        "redCards": agg["totals"].get("redCards", 0),
        "seasonRating": season_rating,
        "behaviorCoefficient": behavior,
    }

    prompt = build_report_prompt(profile, stats, metrics)
    raw = gemini_call(prompt, expect_json=True, system=SCOUT_REPORT_SYSTEM)

    try:
        report = json.loads(raw)
    except:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1])
        try: report = json.loads(cleaned)
        except: return {"error": "Raportul nu a putut fi generat."}

    report["meta"] = {
        "playerId": player_id,
        "playerName": profile.get("name"),
        "matchesAnalyzed": agg["matches_played"],
        "seasonRating": season_rating,
        "behaviorCoefficient": behavior,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
    }
    _REPORT_CACHE[player_id] = report
    return report


# ── REQUEST MODELS ───────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 30


class AdvancedSearchRequest(BaseModel):
    query: Optional[str] = ""
    role_code: Optional[str] = None
    nationality: Optional[str] = None
    preferred_foot: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    min_height_cm: Optional[int] = None
    min_market_value_eur: Optional[int] = None
    max_market_value_eur: Optional[int] = None
    min_matches: Optional[int] = None
    min_minutes: Optional[int] = None
    min_goals: Optional[int] = None
    min_assists: Optional[int] = None
    min_pass_percent: Optional[float] = None
    min_growth_potential: Optional[float] = None
    sort_by_growth: Optional[bool] = False
    limit: Optional[int] = 50


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []


class CompareRequest(BaseModel):
    playerIds: List[int]


# ── ENDPOINTS ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "ScoutIQ U Cluj API v3"}


@app.get("/api/players/{player_id}")
def get_player_profile(player_id: int):
    dto = player_profile_dto(player_id)
    if not dto: raise HTTPException(404, "Player not found")
    return dto


@app.post("/api/search")
def search_players(req: SearchRequest):
    query = req.query.strip()
    if not query:
        return {"results": [], "ai_summary": "", "filters": {}, "search_type": "none"}

    if is_name_query(query):
        name_results = search_by_name(query, max_results=req.limit or 10)
        if name_results:
            return {
                "query": query, "filters": {"name_search": query},
                "count": len(name_results), "results": name_results,
                "ai_summary": f"Am găsit {len(name_results)} jucător{'i' if len(name_results)>1 else ''} care se potrivesc cu '{query}'.",
                "search_type": "name",
            }

    filters = extract_filters_from_query(query)
    results = retrieve_players(filters, query=query, limit=req.limit or 30)

    ai_summary = ""
    if results:
        top5 = results[:5]
        lines = [f"- {p['name']} ({p['position']}, {p['age']} ani, {p['club']}): {p['goals']}G/{p['assists']}A în {p['matches']} meciuri, rating {p['rating']}, {p['marketValue']}" for p in top5]
        prompt = f"""Userul caută: "{query}"

Aceștia sunt SINGURII jucători găsiți în baza de date (date reale):
{chr(10).join(lines)}

Scrie 2-3 fraze despre acești jucători EXACT. Nu inventa alți jucători."""
        ai_summary = gemini_call(prompt, system=SUMMARY_SYSTEM)

    return {
        "query": query, "filters": filters,
        "count": len(results), "results": results,
        "ai_summary": ai_summary, "search_type": "filter",
    }


@app.post("/api/search/advanced")
def search_advanced(req: AdvancedSearchRequest):
    filters = {}
    fields = ["role_code","nationality","preferred_foot","age_min","age_max",
              "min_height_cm","min_market_value_eur","max_market_value_eur",
              "min_matches","min_goals","min_assists","min_pass_percent",
              "min_growth_potential","sort_by_growth"]
    for f in fields:
        v = getattr(req, f, None)
        if v is not None and v != "":
            filters[f] = v

    if req.query and req.query.strip():
        text_filters = extract_filters_from_query(req.query)
        for k, v in text_filters.items():
            if v is not None and k not in filters:
                filters[k] = v

    results = retrieve_players(filters, query=req.query, limit=req.limit or 50)
    return {"query": req.query or "", "filters": filters, "count": len(results), "results": results}


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        return {"reply": "Salut! Sunt asistentul tău de scouting. Ce jucător vrei să analizăm?", "players": []}

    # ── 1. Extrage numele din mesaj dacă e întrebare despre un jucător specific ──
    name_from_context = None
    msg_lower = msg.lower()

    for pattern in ["despre jucătorul nostru, ", "despre jucătorul ", "despre ", "informatii despre ", "informații despre "]:
        if pattern in msg_lower:
            after = msg[msg_lower.index(pattern) + len(pattern):].strip()
            after = after.rstrip("?.!,")
            if after and len(after.split()) <= 4:
                name_from_context = after
                break

    if name_from_context:
        nr = search_by_name(name_from_context, max_results=3)
        if nr:
            top = nr[0]
            prompt = f"""Userul a întrebat: "{msg}"

DATE REALE din baza de date despre {top['name']}:
- Club: {top['club']}, Vârstă: {top['age']} ani, Poziție: {top['position']}
- Sezon: {top['goals']} goluri, {top['assists']} asisturi în {top['matches']} meciuri
- Rating: {top['rating']}/10, Valoare: {top['marketValue']}
- Coeficient comportament: {top['behaviorCoefficient']}/100
- Growth Potential: {top['growthPotential']}/100

Instrucțiuni STRICTE:
- Vorbește NUMAI despre {top['name']} folosind datele de mai sus.
- NU sugera alți jucători, NU inventa statistici.
- 2-3 fraze concise."""
            return {"reply": gemini_call(prompt, system=CHAT_SYSTEM), "players": [top]}

    # ── 2. Căutare directă după nume ──
    if is_name_query(msg):
        nr = search_by_name(msg, max_results=3)

        if nr:
            top = nr[0]
            others_list = nr[1:] if len(nr) > 1 else []

            players_context = (
                f"Jucător: {top['name']}, {top['age']} ani, {top['club']}. "
                f"Stats sezon: {top['goals']} goluri, {top['assists']} asisturi "
                f"în {top['matches']} meciuri, rating {top['rating']}/10, "
                f"valoare {top['marketValue']}, comportament {top['behaviorCoefficient']}/100, "
                f"growth potential {top['growthPotential']}/100."
            )
            if others_list:
                others_str = ", ".join([f"{p['name']} ({p['club']})" for p in others_list])
                players_context += f" Alte rezultate cu același nume: {others_str}."

            prompt = f"""Userul a căutat: "{msg}"

DATE REALE din baza de date:
{players_context}

Instrucțiuni STRICTE:
- Prezintă DOAR pe {top['name']} cu cifrele exacte de mai sus.
- NU adăuga alți jucători care nu sunt în datele de mai sus.
- NU inventa statistici sau context suplimentar.
- 2-3 fraze maxim."""

            return {"reply": gemini_call(prompt, system=CHAT_SYSTEM), "players": nr}

        else:
            return {
                "reply": f"Nu am găsit niciun jucător cu numele '{msg}' în baza de date Liga 1 2024/25. Verifică ortografia sau încearcă numele complet.",
                "players": []
            }

    # ── 3. Detectare intent căutare după criterii ──
    intent_prompt = f'Mesaj: "{msg}"\nE cerere de căutare jucători după criterii (poziție, vârstă, goluri etc.)? JSON: {{"is_search": true/false}}'
    try:
        intent = json.loads(gemini_call(intent_prompt, expect_json=True))
    except:
        intent = {"is_search": False}

    if intent.get("is_search"):
        filters = extract_filters_from_query(msg)
        results = retrieve_players(filters, query=msg, limit=10)

        if not results:
            return {
                "reply": "Nu am găsit jucători cu aceste criterii în baza de date. Încearcă alte filtre — modifică vârsta, poziția sau pragul de goluri.",
                "players": []
            }

        top3 = results[:3]
        lines = [
            f"- {p['name']} ({p['position']}, {p['age']} ani, {p['club']}): "
            f"{p['goals']}G/{p['assists']}A în {p['matches']} meciuri, "
            f"rating {p['rating']}, valoare {p['marketValue']}"
            for p in top3
        ]

        prompt = f"""Userul a căutat: "{msg}"

ACEȘTIA SUNT SINGURII {len(results)} jucători găsiți în baza de date:
{chr(10).join(lines)}

Instrucțiuni STRICTE:
- Menționează NUMAI jucătorii listați mai sus — nu inventa alții.
- Spune că s-au găsit {len(results)} jucători total.
- Evidențiază cel cu ratingul cel mai mare și cel mai bun raport preț/calitate din lista de mai sus.
- 3-4 fraze, ton conversațional de scout."""

        return {"reply": gemini_call(prompt, system=CHAT_SYSTEM), "players": top3}

    # ── 4. Chat general ──
    prompt = f"""Mesaj utilizator: "{msg}"

Răspunde în 1-2 fraze ca asistent de scouting U Cluj.
Dacă întreabă despre un jucător specific, roagă-l să scrie numele direct.
Dacă e conversație generală, fii prietenos și scurt.
NU inventa date sau jucători."""

    return {"reply": gemini_call(prompt, system=CHAT_SYSTEM), "players": []}


@app.get("/api/player/{player_id}/scout-report")
def get_scout_report(player_id: int):
    report = generate_scout_report(player_id)
    if "error" in report: raise HTTPException(404, report["error"])
    return report


@app.post("/api/compare")
def compare_players(req: CompareRequest):
    if len(req.playerIds) < 2:
        raise HTTPException(400, "Minimum 2 players")
    pd = [player_search_dto(pid) for pid in req.playerIds]
    pd = [p for p in pd if p]
    if len(pd) < 2: raise HTTPException(404, "Could not load players")

    lines = [f"- {p['name']} ({p['position']}, {p['age']} ani, {p['club']}): {p['goals']}G/{p['assists']}A, rating {p['rating']}, comportament {p['behaviorCoefficient']}/100, growth potential {p['growthPotential']}/100, {p['marketValue']}" for p in pd]
    prompt = f"Compară NUMAI acești jucători:\n{chr(10).join(lines)}\n\n3 fraze: punctele forte + cel mai bun pentru U Cluj + cel mai bun preț. Nu inventa alți jucători."
    return {"players": pd, "summary": gemini_call(prompt, system=SUMMARY_SYSTEM)}


@app.get("/api/player/{player_id}/similar")
def find_similar(player_id: int, top_n: int = 5):
    target = get_player_doc(player_id)
    if not target: raise HTTPException(404, "Player not found")
    target_dto = player_search_dto(player_id)
    if not target_dto: raise HTTPException(404, "No stats")

    target_role = target.get("role", {}).get("code3", "MID")

    def vec(d):
        m = max(d.get("matches", 1), 1)
        return [d.get("goals",0)/m, d.get("assists",0)/m,
                d.get("rating",0) or 0, d.get("behaviorCoefficient",0) or 0]

    tv = np.array(vec(target_dto))
    similar = []
    for d in db.collection("players").where("role.code3", "==", target_role).stream():
        p = d.to_dict()
        if p["wyId"] == player_id: continue
        dto = player_search_dto(p["wyId"])
        if not dto or dto.get("matches", 0) < 5: continue
        v = np.array(vec(dto))
        n = np.linalg.norm(tv) * np.linalg.norm(v) + 1e-9
        dto["similarity"] = round(float(np.dot(tv, v) / n), 3)
        similar.append(dto)

    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return {"target": target_dto, "similar": similar[:top_n]}


@app.get("/api/bargains")
def find_bargains(role: Optional[str] = None, max_value: int = 500_000, top_n: int = 10):
    q = db.collection("players")
    if role: q = q.where("role.code3", "==", role)
    bargains = []
    for d in q.stream():
        p = d.to_dict()
        mv = p.get("marketValueEur", 0) or 0
        if mv == 0 or mv > max_value: continue
        dto = player_search_dto(p["wyId"])
        if not dto or dto.get("matches", 0) < 5: continue
        rating = dto.get("rating") or 0
        dto["bargainScore"] = round(rating / (mv/100_000 + 1), 2)
        bargains.append(dto)
    bargains.sort(key=lambda x: x["bargainScore"], reverse=True)
    return {"bargains": bargains[:top_n]}


@app.post("/api/cache/clear")
def clear_cache():
    global _PLAYER_CACHE, _STATS_CACHE, _REPORT_CACHE, _ALL_PLAYERS_CACHE
    _PLAYER_CACHE = {}; _STATS_CACHE = {}; _REPORT_CACHE = {}; _ALL_PLAYERS_CACHE = None
    return {"status": "cache cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)