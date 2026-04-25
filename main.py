"""
ScoutIQ U Cluj — FastAPI Backend (v2 — rating realist + chat strict)
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

GEMINI_API_KEY = "REMOVED_REVOKED_CREDENTIAL"
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-3-flash-preview" # gemini-3-flash-preview, gemini-2.5-flash 

app = FastAPI(title="ScoutIQ U Cluj API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def calculate_age(birth_date: str) -> int:
    try:
        return datetime.now().year - int(birth_date[:4])
    except:
        return 0


def fmt_market_value(eur: int) -> str:
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
    full_prompt = prompt
    if system:
        full_prompt = f"{system}\n\n{prompt}"
    response = client.models.generate_content(model=MODEL, contents=full_prompt)
    text = response.text.strip()
    if expect_json:
        text = text.replace("```json", "").replace("```", "").strip()
    return text


# ════════════════════════════════════════════════════════════════
# ANTI-HALLUCINATION — sistem mai strict (v2)
# ════════════════════════════════════════════════════════════════

ANTI_HALLUCINATION_SYSTEM = """Ești un asistent strict de scouting pentru FC Universitatea Cluj.

REGULI ABSOLUTE — orice abatere este interzisă:
1. Folosești EXCLUSIV datele furnizate în prompt. Zero invenții.
2. Dacă un câmp e "—", null sau lipsește → scrie "informație indisponibilă". Nu ghici.
3. Orice cifră din răspuns trebuie să existe VERBATIM în datele primite.
4. Nu menționezi jucători care nu apar în lista furnizată.
5. Nu inventa cluburi, contracte, naționalități sau prețuri.
6. Răspunde EXCLUSIV în română, scurt și profesionist — max 4 fraze.
7. Nu face comparații cu jucători celebri (Messi, Ronaldo etc.) — ești scout, nu comentator.
8. Dacă datele sunt insuficiente pentru o concluzie, spune explicit asta.
9. Nu folosi expresii vagi ca "jucător talentat", "potențial enorm" fără cifre concrete.
"""

CHAT_SYSTEM = """Ești un asistent de scouting pentru FC Universitatea Cluj.
Ai acces DOAR la datele din baza de date furnizate mai jos.

REGULI STRICTE:
1. Răspunde DOAR pe baza datelor furnizate — nicio inventie.
2. Fiecare afirmație despre un jucător trebuie susținută de o cifră din date.
3. Dacă nu ai date suficiente, spune "Nu am suficiente date pentru această întrebare."
4. Răspunsuri scurte: 2-3 fraze maxim.
5. Nu recomanda jucători care nu sunt în lista furnizată.
6. Nu face comparații cu jucători celebri sau din alte ligi.
7. La final, sugerează utilizatorului să acceseze profilul complet pentru detalii.
"""


# ════════════════════════════════════════════════════════════════
# DATA AGGREGATION
# ════════════════════════════════════════════════════════════════

def aggregate_player_stats(player_id: int) -> dict:
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

        if minutes < 15:
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


# ════════════════════════════════════════════════════════════════
# METRICA #1 — SEASON RATING (v2 — realist, fără normalizare pe minut)
# ════════════════════════════════════════════════════════════════

# Ponderi per rol — acțiuni pozitive și penalizări negative
ROLE_WEIGHTS = {
    "MID": {
        # Ofensiv
        "goals": 1.2,
        "assists": 0.8,
        "successfulKeyPasses": 0.15,
        "successfulPasses": 0.02,
        "passes": -0.03,                        # pase ratate (total - successful nu e disponibil direct)
        "successfulDribbles": 0.1,
        "dribbles": -0.04,                      # driblinguri ratate
        "xgShot": 0.1,
        "xgAssist": 0.08,
        # Defensiv
        "interceptions": 0.05,
        "successfulSlidingTackles": 0.1,
        "slidingTackles": -0.05,               # tackle-uri ratate
        "recoveries": 0.04,
        # Disciplină și pierderi
        "yellowCards": -0.3,
        "dangerousOwnHalfLosses": -0.25,
        "losses": -0.04,
    },
    "FWD": {
        "goals": 1.2,
        "assists": 0.8,
        "successfulDribbles": 0.1,
        "dribbles": -0.04,
        "touchInBox": 0.05,
        "xgShot": 0.15,
        "xgAssist": 0.08,
        "successfulKeyPasses": 0.1,
        "shotsOnTarget": 0.1,
        "successfulPasses": 0.015,
        "yellowCards": -0.3,
        "offsides": -0.1,
        "losses": -0.04,
        "dangerousOwnHalfLosses": -0.2,
    },
    "DEF": {
        "defensiveDuelsWon": 0.12,
        "aerialDuelsWon": 0.12,
        "interceptions": 0.12,
        "clearances": 0.08,
        "successfulSlidingTackles": 0.1,
        "slidingTackles": -0.05,
        "recoveries": 0.06,
        "successfulProgressivePasses": 0.05,
        "successfulPasses": 0.015,
        "goals": 0.8,
        "assists": 0.6,
        "dangerousOwnHalfLosses": -0.5,
        "yellowCards": -0.3,
        "losses": -0.05,
    },
    "GKP": {
        "gkSaves": 0.3,
        "gkCleanSheets": 0.8,
        "gkSuccessfulExits": 0.1,
        "gkAerialDuelsWon": 0.08,
        "successfulLongPasses": 0.04,
        "successfulPasses": 0.01,
        "gkConcededGoals": -0.4,
        "yellowCards": -0.3,
    },
}

# Map din role_code Wyscout → categoria de ponderi
ROLE_MAP = {
    "MID": "MID", "AMD": "MID", "DMD": "MID",
    "FWD": "FWD", "AMF": "FWD",
    "DEF": "DEF", "LB": "DEF", "RB": "DEF", "CB": "DEF",
    "GKP": "GKP", "GK": "GKP",
}


def get_role_weights(role_code: str) -> dict:
    return ROLE_WEIGHTS.get(ROLE_MAP.get(role_code, "MID"), ROLE_WEIGHTS["MID"])


def calculate_match_rating(total_stats: dict, role_code: str, minutes: int) -> Optional[float]:
    """
    Rating per meci — v2 (inspirat din Opta/SofaScore):
    - Pornire de la 6.0 (standard industrie)
    - Impact direct al acțiunilor fără normalizare pe minut
    - Modificatori contextuali: roșu, erori la gol, clean sheet
    """
    has_major_impact = (
        total_stats.get("goals", 0) > 0
        or total_stats.get("assists", 0) > 0
        or total_stats.get("redCards", 0) > 0
    )

    # Jucătorii cu sub 15 minute primesc rating doar dacă au impact major
    if minutes < 15 and not has_major_impact:
        return None

    rating = 6.0

    weights = get_role_weights(role_code)

    # Impact direct al acțiunilor (fără normalizare pe minut — exact ca Opta)
    action_impact = sum(
        (total_stats.get(stat, 0) or 0) * weight
        for stat, weight in weights.items()
    )
    rating += action_impact

    # ── Modificatori contextuali ─────────────────────────────

    # 1. Bonus clean sheet (portari/fundași care joacă ≥60 min)
    if minutes >= 60 and total_stats.get("gkConcededGoals", 1) == 0:
        if role_code in ("GKP", "GK", "DEF", "CB", "LB", "RB"):
            rating += 0.5
        elif role_code in ("DMD",):
            rating += 0.2

    # 2. Penalizare majoră cartonaș roșu (max 5.5 cu roșu)
    if total_stats.get("redCards", 0) > 0:
        rating = min(rating, 5.5)

    # 3. Penalizare erori care duc la gol
    errors_to_goal = total_stats.get("dangerousOwnHalfLosses", 0) or 0
    if errors_to_goal > 0:
        rating -= 0.8 * errors_to_goal

    return round(max(1.0, min(10.0, rating)), 2)


def calculate_season_rating_for_player(player_id: int, role_code: str) -> Optional[float]:
    agg = aggregate_player_stats(player_id)
    if not agg:
        return None
    ratings = [
        calculate_match_rating(m["totalRaw"], role_code, m["minutes"])
        for m in agg["match_history"]
    ]
    ratings = [r for r in ratings if r is not None]
    if not ratings:
        return None
    return round(sum(ratings) / len(ratings), 2)


# ════════════════════════════════════════════════════════════════
# METRICA #2 — BEHAVIOR COEFFICIENT
# ════════════════════════════════════════════════════════════════

def calculate_behavior_coefficient(totals: dict, matches: int, role_code: str) -> float:
    if matches == 0:
        return 0.0

    score = 50.0

    # Disciplină (max -20)
    yellow_per_match = (totals.get("yellowCards", 0) or 0) / matches
    red_per_match = (totals.get("redCards", 0) or 0) / matches
    fouls_per_match = (totals.get("fouls", 0) or 0) / matches
    discipline_penalty = (yellow_per_match * 8) + (red_per_match * 25) + (fouls_per_match * 1.5)
    score -= min(discipline_penalty, 20)

    # Consistență minute (max +15)
    minutes_per_match = (totals.get("minutesOnField", 0) or 0) / matches
    score += min((minutes_per_match / 90) * 15, 15)

    # Eficiență acțiuni (max +20)
    pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0)) or 0
    duels_pct = safe_pct(totals.get("duelsWon", 0), totals.get("duels", 0)) or 0
    score += min((pass_acc * 0.15) + (duels_pct * 0.25), 20)

    # Pierderi periculoase (max -15)
    dangerous_per_match = (totals.get("dangerousOwnHalfLosses", 0) or 0) / matches
    score -= min(dangerous_per_match * 5, 15)

    return round(max(0.0, min(100.0, score)), 1)


# ════════════════════════════════════════════════════════════════
# METRICA #3 — COMPATIBILITY SCORE
# ════════════════════════════════════════════════════════════════

def calculate_compatibility(totals: dict, matches: int, role_code: str, query_filters: dict) -> float:
    if not query_filters or matches == 0:
        return 50.0

    checks = 0
    hits = 0

    criteria = [
        ("min_matches", lambda: matches >= query_filters["min_matches"]),
        ("min_goals", lambda: totals.get("goals", 0) >= query_filters["min_goals"]),
        ("min_assists", lambda: totals.get("assists", 0) >= query_filters["min_assists"]),
        ("min_pass_percent", lambda: (safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0)) or 0) >= query_filters["min_pass_percent"]),
    ]

    for key, check_fn in criteria:
        if query_filters.get(key):
            checks += 1
            if check_fn():
                hits += 1

    if checks == 0:
        return 75.0

    base = (hits / checks) * 70
    pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0)) or 0
    quality_bonus = min(pass_acc / 100 * 30, 30)
    return round(min(100.0, base + quality_bonus), 1)

"""
Scout Report — adăugat în main_improved.py
==========================================

1. Adaugă funcția `generate_scout_report()` și `score_domain()`
2. Adaugă endpoint-ul  GET /api/player/{player_id}/scout-report
3. (Opțional) Adaugă câmpul `scoutReport` în player_profile_dto()
   dacă vrei să fie încărcat o dată cu profilul.

Integrare în main_improved.py:
  - Copiază SCOUT_REPORT_SYSTEM și funcțiile de mai jos
  - Adaugă endpoint-ul în secțiunea ENDPOINTS
"""

# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — strict, structurat
# ════════════════════════════════════════════════════════════════

SCOUT_REPORT_SYSTEM = """Ești un scout profesionist pentru FC Universitatea Cluj.
Ai primit datele statistice REALE ale unui jucător pentru un sezon întreg.

REGULI ABSOLUTE:
1. Folosești EXCLUSIV cifrele furnizate. Zero invenții.
2. Fiecare afirmație trebuie susținută de o cifră concretă din date.
3. Dacă un câmp lipsește sau e 0, nu-l menciona ca punct forte.
4. Nu compara cu jucători celebri (Messi, Ronaldo, Modric etc.).
5. Nu folosi superlative fără cifre: "extraordinar", "fenomenal" — interzis.
6. Răspunde EXCLUSIV în română, ton profesionist de raport de scouting.
7. Returnează DOAR JSON valid, fără markdown, fără text în afara JSON-ului.
"""

SCOUT_REPORT_PROMPT_TEMPLATE = """Date jucător (sezon complet, REALE):
Nume: {name}
Poziție principală: {position}
Vârstă: {age} ani
Club: {club}
Valoare piață: {market_value}
Picior preferat: {foot}
Înălțime: {height}

Statistici sezon ({matches} meciuri, {minutes} minute):
- Goluri: {goals} | Asisturi: {assists}
- xG: {xg} | xA: {xa}
- Pase totale: {passes_total} | Pase reușite: {passes_ok} | Acuratețe: {pass_acc}%
- Pase cheie: {key_passes}
- Pase lungi reușite: {long_passes_ok} / {long_passes_total}
- Pase progresive: {progressive_passes}
- Driblinguri reușite: {dribbles_ok} / {dribbles_total}
- Dueluri câștigate: {duels_won} / {duels_total} ({duels_pct}%)
- Dueluri aeriene: {aerial_won} / {aerial_total} ({aerial_pct}%)
- Intercepții: {interceptions}
- Recuperări: {recoveries}
- Degajări: {clearances}
- Faulturi comise: {fouls_committed} | Faulturi primite: {fouls_received}
- Galbene: {yellow_cards} | Roșii: {red_cards}
- Pierderi periculoase: {dangerous_losses}

Metrici calculate:
- Rating sezon: {season_rating}/10
- Coeficient comportament: {behavior}/100
- Compatibilitate cu profilul căutat: {compatibility}%

Generează un raport de scouting complet în formatul JSON de mai jos.
Returnează DOAR JSON-ul, fără nicio altă explicație sau text:

{{
  "verdict": {{
    "text": "Un paragraf de 3-5 fraze cu evaluarea generală. Fiecare afirmație să aibă cifre concrete.",
    "recommendation": "Transfer recomandat / Transfer fezabil / Observare suplimentară / Nu se recomandă",
    "badges": ["max 3 badge-uri scurte: ex: Mijlocaș echilibru, Valoare bună, Profil tânăr"]
  }},
  "domains": {{
    "constructie": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "faza_ofensiva": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "faza_defensiva": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "dueluri_fizice": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }},
    "disciplina": {{ "score": 0-10, "comment": "max 15 cuvinte cu cifre" }}
  }},
  "strengths": [
    "Punct forte 1 cu cifre exacte",
    "Punct forte 2 cu cifre exacte",
    "Punct forte 3 cu cifre exacte"
  ],
  "weaknesses": [
    "Punct slab 1 cu cifre exacte",
    "Punct slab 2 cu cifre exacte"
  ],
  "scout_recommendation": "2-3 fraze despre utilizarea jucătorului la U Cluj, rolul ideal, condiții de transfer. Cu cifre."
}}"""


# ════════════════════════════════════════════════════════════════
# GENERARE RAPORT
# ════════════════════════════════════════════════════════════════

def build_scout_report_prompt(profile: dict, stats: dict, metrics: dict) -> str:
    """Construiește prompt-ul cu datele reale, fără a lăsa câmpuri goale."""

    def pct(a, b):
        if not b or b == 0:
            return 0
        return round(a / b * 100, 1)

    return SCOUT_REPORT_PROMPT_TEMPLATE.format(
        name=profile.get("name", "—"),
        position=profile.get("position", "—"),
        age=profile.get("age", "—"),
        club=profile.get("club", "—"),
        market_value=profile.get("marketValue", "—"),
        foot=profile.get("preferredFoot", "—"),
        height=profile.get("height", "—"),

        matches=stats.get("matches", 0),
        minutes=stats.get("minutes", 0),
        goals=stats.get("goals", 0),
        assists=stats.get("assists", 0),
        xg=stats.get("xG", 0),
        xa=stats.get("xA", 0),

        passes_total=stats.get("pasesTotal", 0),
        passes_ok=stats.get("successfulPasses", 0),
        pass_acc=stats.get("pasesAccuracy") or 0,
        key_passes=stats.get("pasesKey", 0),
        long_passes_ok=stats.get("pasesLong", 0),
        long_passes_total=stats.get("pasesLong", 0) + 1,   # fallback safe
        progressive_passes=stats.get("pasesProgressive", 0),

        dribbles_ok=stats.get("driblings", 0),
        dribbles_total=stats.get("driblingAttempts", 0),

        duels_won=stats.get("duelsWon", 0),
        duels_total=stats.get("duelsTotal", 0),
        duels_pct=pct(stats.get("duelsWon", 0), stats.get("duelsTotal", 0)),

        aerial_won=stats.get("aerialWon", 0),
        aerial_total=stats.get("aerialDuels", 0),
        aerial_pct=pct(stats.get("aerialWon", 0), stats.get("aerialDuels", 0)),

        interceptions=stats.get("interceptions", 0),
        recoveries=stats.get("recoveries", 0),
        clearances=stats.get("clearances", 0),

        fouls_committed=stats.get("foulsCommitted", 0),
        fouls_received=stats.get("foulsReceived", 0),
        yellow_cards=int(metrics.get("yellowCards", 0)),
        red_cards=int(metrics.get("redCards", 0)),
        dangerous_losses=stats.get("errorsToShot", 0),

        season_rating=metrics.get("seasonRating", "—"),
        behavior=metrics.get("behaviorCoefficient", "—"),
        compatibility=metrics.get("compatibilityScore", "N/A"),
    )


def generate_scout_report(player_id: int) -> dict:
    """
    Generează raportul de scouting complet pentru un jucător.
    Returnează dict cu structura validată.
    """
    import json as _json

    # 1. Încărcăm profilul complet
    profile = player_profile_dto(player_id)
    if not profile:
        return {"error": "Jucătorul nu a fost găsit sau nu are statistici."}

    stats = profile.get("stats", {})

    # 2. Calculăm metricile extra (cartonașe din totals)
    agg = aggregate_player_stats(player_id)
    if not agg:
        return {"error": "Date insuficiente pentru raport."}

    totals = agg["totals"]
    role_code = "MID"  # fallback — în practică vine din player doc
    p_doc = db.collection("players").document(str(player_id)).get()
    if p_doc.exists:
        role_code = p_doc.to_dict().get("role", {}).get("code3", "MID")

    behavior = calculate_behavior_coefficient(totals, agg["matches_played"], role_code)
    season_rating = calculate_season_rating_for_player(player_id, role_code)

    metrics = {
        "yellowCards": totals.get("yellowCards", 0),
        "redCards": totals.get("redCards", 0),
        "seasonRating": season_rating,
        "behaviorCoefficient": behavior,
        "compatibilityScore": None,  # nu avem query context aici
    }

    # 3. Construim prompt-ul strict
    prompt = build_scout_report_prompt(profile, stats, metrics)

    # 4. Apel Gemini — forțăm JSON pur
    raw = gemini_call(prompt, expect_json=True, system=SCOUT_REPORT_SYSTEM)

    # 5. Parsăm și validăm
    try:
        report = _json.loads(raw)
    except Exception:
        # Fallback: dacă Gemini tot întoarce markdown, curățăm agresiv
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        try:
            report = _json.loads(cleaned)
        except Exception:
            return {"error": "Raportul nu a putut fi generat. Încearcă din nou.", "raw": raw[:300]}

    # 6. Adăugăm metadatele calculat (cifre din cod, nu din Gemini)
    report["meta"] = {
        "playerId": player_id,
        "playerName": profile.get("name"),
        "matchesAnalyzed": agg["matches_played"],
        "seasonRating": season_rating,
        "behaviorCoefficient": behavior,
        "generatedAt": datetime.utcnow().isoformat() + "Z",
    }

    return report


# ════════════════════════════════════════════════════════════════
# ENDPOINT — adaugă în secțiunea ENDPOINTS din main_improved.py
# ════════════════════════════════════════════════════════════════

@app.get("/api/player/{player_id}/scout-report")
def get_scout_report(player_id: int):
    """
    Generează un raport de scouting detaliat pentru un jucător.
    Folosit de tab-ul 'Scout Report' din PlayerProfile.js

    Response shape:
    {
      "verdict": { "text": "...", "recommendation": "...", "badges": [...] },
      "domains": {
        "constructie": { "score": 7.5, "comment": "..." },
        "faza_ofensiva": { ... },
        "faza_defensiva": { ... },
        "dueluri_fizice": { ... },
        "disciplina": { ... }
      },
      "strengths": ["...", "...", "..."],
      "weaknesses": ["...", "..."],
      "scout_recommendation": "...",
      "meta": {
        "playerId": 123,
        "playerName": "...",
        "matchesAnalyzed": 24,
        "seasonRating": 7.4,
        "behaviorCoefficient": 81.0,
        "generatedAt": "2025-..."
      }
    }
    """
    report = generate_scout_report(player_id)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return report


# ════════════════════════════════════════════════════════════════
# (OPȚIONAL) Integrare în player_profile_dto
# Dacă vrei raportul preîncărcat o dată cu profilul (mai lent),
# adaugă la finalul funcției player_profile_dto():
#
#   dto["scoutReport"] = generate_scout_report(player_id)
#   return dto
#
# Recomandat: NU face asta — generarea durează 2-4s.
# Lasă frontend-ul să apeleze /scout-report separat,
# la click pe tab, cu loading state.
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# PLAYER DTOs
# ════════════════════════════════════════════════════════════════

def player_profile_dto(player_id: int) -> dict:
    p_doc = db.collection("players").document(str(player_id)).get()
    if not p_doc.exists:
        return None
    p = p_doc.to_dict()

    agg = aggregate_player_stats(player_id)
    if not agg:
        return None

    totals = agg["totals"]
    matches = agg["matches_played"]

    pass_acc = safe_pct(totals.get("successfulPasses", 0), totals.get("passes", 0))
    cross_acc = safe_pct(totals.get("successfulCrosses", 0), totals.get("crosses", 0))
    long_acc = safe_pct(totals.get("successfulLongPasses", 0), totals.get("longPasses", 0))

    role_code = p.get("role", {}).get("code3", "MID")
    season_ratings = []
    for i, m in enumerate(agg["match_history"]):
        rating = calculate_match_rating(m["totalRaw"], role_code, m["minutes"])
        if rating is not None:
            label = m["matchLabel"]
            opponent = "Necunoscut"
            try:
                parts = label.split(",")[0]
                teams = parts.split(" - ")
                opponent = teams[1].strip() if len(teams) > 1 else "Necunoscut"
            except:
                pass
            season_ratings.append({
                "matchday": f"Et.{i+1}",
                "rating": rating,
                "opponent": opponent,
                "home": True,
            })

    overall_rating = round(
        sum(r["rating"] for r in season_ratings) / max(len(season_ratings), 1), 1
    ) if season_ratings else None

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
            "pasesShortAcc": pass_acc,
            "pasesMedium": int(totals.get("passes", 0)) // 3,
            "pasesMediumAcc": pass_acc,
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

    # Position code normalizat
    position_code = role_code
    main_pos = agg["main_position"] or ""
    pos_map = {
        "Centre Back": "CB", "Center Back": "CB",
        "Left Back": "LB", "Right Back": "RB",
        "Striker": "ST", "Centre Forward": "ST",
        "Right Wing": "RW", "Left Wing": "LW",
        "Defensive Midfielder": "CDM",
        "Attacking Midfielder": "CAM",
        "Centre Midfielder": "CM", "Center Midfielder": "CM",
        "Goalkeeper": "GK",
    }
    for key, code in pos_map.items():
        if key in main_pos:
            position_code = code
            break

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
        "rating": season_rating,
        "behaviorCoefficient": behavior,
        "compatibilityScore": compatibility,
        "status": "monitorizat",
    }


# ════════════════════════════════════════════════════════════════
# RAG — RETRIEVE & FILTER
# ════════════════════════════════════════════════════════════════

def extract_filters_from_query(query: str) -> dict:
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

        age = calculate_age(p.get("birthDate", ""))
        if filters.get("age_max") and age > filters["age_max"]:
            continue
        if filters.get("age_min") and age < filters["age_min"]:
            continue
        if filters.get("min_height_cm") and (p.get("height", 0) or 0) < filters["min_height_cm"]:
            continue

        mv = p.get("marketValueEur", 0) or 0
        if filters.get("max_market_value_eur") and mv > 0 and mv > filters["max_market_value_eur"]:
            continue
        if filters.get("min_market_value_eur") and mv < filters["min_market_value_eur"]:
            continue

        agg = aggregate_player_stats(p["wyId"])
        if not agg:
            continue

        totals = agg["totals"]
        matches = agg["matches_played"]

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

        dto = player_search_dto(p["wyId"], query_context=filters)
        if dto:
            results.append(dto)

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
    dto = player_profile_dto(player_id)
    if not dto:
        raise HTTPException(status_code=404, detail="Player not found")
    return dto


@app.post("/api/search")
def search_players(req: SearchRequest):
    query = req.query.strip()
    if not query:
        return {"results": [], "ai_summary": "", "filters": {}}

    filters = extract_filters_from_query(query)
    results = retrieve_players(filters, query=query, limit=req.limit or 30)

    ai_summary = ""
    if results:
        top5 = results[:5]
        # Construim un rezumat strict cu cifrele reale
        stats_lines = []
        for pl in top5:
            stats_lines.append(
                f"- {pl['name']} ({pl['position']}, {pl['age']} ani, {pl['club']}): "
                f"{pl['goals']} goluri, {pl['assists']} asisturi în {pl['matches']} meciuri, "
                f"rating {pl['rating']}, valoare {pl['marketValue']}"
            )
        prompt = f"""Userul caută: "{query}"

Jucători găsiți în baza de date (date reale, nu adăuga altceva):
{chr(10).join(stats_lines)}

Scrie un rezumat scurt în 3 fraze:
1. Câți jucători s-au găsit și ce poziție au
2. Cel mai recomandat și DE CE (citează cifre exacte de mai sus)
3. Cel mai accesibil bugetar (dacă diferă)

Folosește EXCLUSIV cifrele de mai sus."""
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
    Chat strict anti-hallucination — v2.
    Datele sunt furnizate explicit în prompt, Gemini nu poate inventa nimic.
    """
    msg = req.message.strip()
    if not msg:
        return {"reply": "Spune-mi ce cauți și caut în baza de date.", "players": []}

    # Detectare intent
    intent_prompt = f"""Mesaj user: "{msg}"
E un mesaj care necesită căutare de jucători? (ex: "atacant tânăr", "fundaș sub 25 ani")
Răspunde DOAR cu JSON: {{"is_search": true/false}}"""
    try:
        intent = json.loads(gemini_call(intent_prompt, expect_json=True))
    except:
        intent = {"is_search": True}

    if intent.get("is_search"):
        filters = extract_filters_from_query(msg)
        results = retrieve_players(filters, query=msg, limit=10)

        if not results:
            return {
                "reply": "Nu am găsit jucători care să corespundă criteriilor tale în baza de date. Încearcă cu alte filtre.",
                "players": []
            }

        top3 = results[:3]

        # Construim datele strict formatate pentru prompt
        player_lines = []
        for pl in top3:
            player_lines.append(
                f"- {pl['name']} ({pl['position']}, {pl['age']} ani, {pl['club']}): "
                f"{pl['goals']} goluri, {pl['assists']} asisturi în {pl['matches']} meciuri, "
                f"rating sezon {pl['rating']}, valoare {pl['marketValue']}, "
                f"coef. comportament {pl['behaviorCoefficient']}"
            )

        prompt = f"""Userul a întrebat: "{msg}"

Date REALE din baza de date (nu adăuga nicio informație suplimentară):
{chr(10).join(player_lines)}

Răspunde în 2-3 fraze:
- Menționează jucătorul recomandat cu cifrele exacte de mai sus
- Explică pe scurt de ce (bazat exclusiv pe cifre)
- Sugerează accesarea profilului pentru detalii complete

INTERZIS: informații care nu apar în datele de mai sus."""

        reply = gemini_call(prompt, system=CHAT_SYSTEM)

        return {"reply": reply, "players": top3}

    else:
        # Conversație generală — răspuns scurt, fără inventii
        prompt = f"""Ești asistent de scouting pentru FC Universitatea Cluj.
Mesaj user: "{msg}"

Răspunde scurt (1-2 fraze). Dacă userul vrea să caute jucători, cere-i să fie specific:
poziție, vârstă, criterii statistice. Nu inventa informații despre jucători sau liga."""
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

    # Formatăm datele strict pentru comparație
    player_lines = []
    for pl in players_data:
        player_lines.append(
            f"- {pl['name']} ({pl['position']}, {pl['age']} ani, {pl['club']}): "
            f"{pl['goals']} goluri, {pl['assists']} asisturi în {pl['matches']} meciuri, "
            f"rating {pl['rating']}, comportament {pl['behaviorCoefficient']}/100, "
            f"valoare {pl['marketValue']}"
        )

    prompt = f"""Compară acești jucători pe baza datelor de mai jos:
{chr(10).join(player_lines)}

Răspunde în română:
1. Punctele forte ale fiecăruia (cu cifre exacte din date)
2. Cel mai bun pentru U Cluj și de ce (bazat pe cifre)
3. Cel mai bun raport calitate/preț

Folosește EXCLUSIV datele furnizate. Nu adăuga informații externe."""

    summary = gemini_call(prompt, system=ANTI_HALLUCINATION_SYSTEM)
    return {"players": players_data, "summary": summary}


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

    def get_vec(dto):
        m = max(dto.get("matches", 1), 1)
        return [
            dto.get("goals", 0) / m,
            dto.get("assists", 0) / m,
            dto.get("rating", 0) or 0,
            dto.get("behaviorCoefficient", 0) or 0,
        ]

    target_vec = np.array(get_vec(target_dto))
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
        norm = np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-9
        sim = float(np.dot(target_vec, vec) / norm)
        dto["similarity"] = round(sim, 3)
        similar.append(dto)

    similar.sort(key=lambda x: x["similarity"], reverse=True)
    return {"target": target_dto, "similar": similar[:top_n]}


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