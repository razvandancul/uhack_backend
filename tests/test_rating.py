import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict

# Setup Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── FORMULA WEIGHTED (bazată pe rol) ─────────────────────────
WEIGHTS = {
    "MID": {
        "goals": 8,
        "assists": 6,
        "xgShot": 5,
        "xgAssist": 4,
        "successfulPasses": 3,
        "successfulDribbles": 3,
        "interceptions": 2,
        "duelsWon": 2,
        "losses": -2,
        "dangerousOwnHalfLosses": -3,
    },
    "FWD": {
        "goals": 10,
        "assists": 6,
        "xgShot": 6,
        "xgAssist": 4,
        "successfulDribbles": 4,
        "touchInBox": 3,
        "losses": -2,
        "offsides": -1,
    },
    "DEF": {
        "defensiveDuelsWon": 6,
        "aerialDuelsWon": 5,
        "interceptions": 5,
        "clearances": 4,
        "assists": 3,
        "goals": 4,
        "dangerousOwnHalfLosses": -5,
        "losses": -2,
    },
    "GKP": {
        "gkSaves": 8,
        "gkCleanSheets": 10,
        "gkSuccessfulExits": 4,
        "gkAerialDuelsWon": 3,
        "gkConcededGoals": -6,
    },
}

def get_weights_for_role(role_code):
    mapping = {
        "MID": "MID", "AMD": "MID", "DMD": "MID",
        "FWD": "FWD", "AMF": "FWD",
        "DEF": "DEF", "LB": "DEF", "RB": "DEF", "CB": "DEF",
        "GKP": "GKP",
    }
    return WEIGHTS.get(mapping.get(role_code, "MID"), WEIGHTS["MID"])

def calculate_match_rating(total_stats: dict, role_code: str, minutes: int) -> float:
    """Rating pentru un singur meci (scală 1-10)"""
    if minutes < 20:
        return None  # prea puține minute pentru rating relevant

    weights = get_weights_for_role(role_code)
    raw_score = 0

    for stat, weight in weights.items():
        value = total_stats.get(stat, 0) or 0
        raw_score += value * weight

    # Normalizăm la 90 minute
    normalized = raw_score * (90 / minutes)

    # Mapăm la scală 1-10 (baseline 6.0)
    rating = 6.0 + (normalized / 10)
    rating = max(1.0, min(10.0, rating))

    return round(rating, 2)

# ── PERCENTILE RATING ─────────────────────────────────────────
def calculate_percentile_ratings(all_players_stats: list, role_code: str) -> dict:
    """
    Calculează percentile față de toți jucătorii din același rol.
    Returnează {playerId: percentile_rating}
    """
    key_stats = {
        "MID": ["goals", "assists", "xgShot", "successfulPasses", "interceptions", "duelsWon"],
        "FWD": ["goals", "assists", "xgShot", "touchInBox", "successfulDribbles"],
        "DEF": ["defensiveDuelsWon", "aerialDuelsWon", "interceptions", "clearances"],
        "GKP": ["gkSaves", "gkCleanSheets", "gkSuccessfulExits"],
    }

    role_mapping = {
        "MID": "MID", "AMD": "MID", "DMD": "MID",
        "FWD": "FWD", "AMF": "FWD",
        "DEF": "DEF", "LB": "DEF", "RB": "DEF", "CB": "DEF",
        "GKP": "GKP",
    }

    stats_keys = key_stats.get(role_mapping.get(role_code, "MID"), key_stats["MID"])

    # Colectăm valorile per jucător
    player_values = {}
    for p in all_players_stats:
        pid = p["playerId"]
        total = p.get("totalSeason", {})
        matches = p.get("matchesPlayed", 1) or 1

        # Normalizăm per 90 minute
        minutes = total.get("minutesOnField", 1) or 1
        player_values[pid] = {
            stat: (total.get(stat, 0) or 0) / minutes * 90
            for stat in stats_keys
        }

    # Calculăm percentile per stat
    percentile_scores = {}
    for pid in player_values:
        score = 0
        for stat in stats_keys:
            values = [player_values[p][stat] for p in player_values]
            player_val = player_values[pid][stat]
            percentile = sum(1 for v in values if v <= player_val) / len(values)
            score += percentile

        # Scală 1-10
        percentile_rating = 1 + (score / len(stats_keys)) * 9
        percentile_scores[pid] = round(percentile_rating, 2)

    return percentile_scores

# ── TEST PE UN JUCĂTOR DIN FIRESTORE ─────────────────────────
def test_single_player(player_id: int):
    print(f"\n{'='*50}")
    print(f"🔍 Testing player {player_id}")
    print(f"{'='*50}")

    # Info jucător
    player_doc = db.collection("players").document(str(player_id)).get()
    if not player_doc.exists:
        print(f"❌ Player {player_id} not found!")
        return

    player = player_doc.to_dict()
    role_code = player.get("role", {}).get("code3", "MID")
    print(f"👤 {player.get('shortName')} | Rol: {role_code} | Echipă ID: {player.get('currentTeamId')}")

    # Statistici per meci
    stats_docs = db.collection("playerStats")\
        .where("playerId", "==", player_id)\
        .stream()

    match_ratings = []
    season_totals = defaultdict(float)
    total_minutes = 0

    print(f"\n📊 Rating per meci:")
    print(f"{'Meci':<45} {'Min':>5} {'Rating':>8}")
    print("-" * 60)

    for doc in stats_docs:
        data = doc.to_dict()
        total = data.get("total", {})
        minutes = total.get("minutesOnField", 0) or 0

        if minutes < 20:
            continue

        rating = calculate_match_rating(total, role_code, minutes)
        match_label = data.get("matchLabel", "Unknown")[:44]

        print(f"{match_label:<45} {minutes:>5} {rating:>8.2f}")

        match_ratings.append(rating)
        total_minutes += minutes

        # Acumulăm pentru sezon
        for key, val in total.items():
            if isinstance(val, (int, float)):
                season_totals[key] += val

    if not match_ratings:
        print("⚠️  Niciun meci cu minute suficiente găsit!")
        return

    # Statistici sezon
    season_avg = round(sum(match_ratings) / len(match_ratings), 2)
    last5_avg = round(sum(match_ratings[-5:]) / min(5, len(match_ratings)), 2)
    best_match = max(match_ratings)
    worst_match = min(match_ratings)

    print(f"\n{'='*50}")
    print(f"📈 RATING SEZON:")
    print(f"  Meciuri jucate:  {len(match_ratings)}")
    print(f"  Minute totale:   {int(total_minutes)}")
    print(f"  Rating mediu:    {season_avg}/10")
    print(f"  Ultimele 5 meci: {last5_avg}/10")
    print(f"  Cel mai bun:     {best_match}/10")
    print(f"  Cel mai slab:    {worst_match}/10")
    print(f"\n📊 STATISTICI CHEIE SEZON:")
    print(f"  Goluri:          {int(season_totals.get('goals', 0))}")
    print(f"  Asisturi:        {int(season_totals.get('assists', 0))}")
    print(f"  xG:              {round(season_totals.get('xgShot', 0), 2)}")
    print(f"  Pase reușite:    {int(season_totals.get('successfulPasses', 0))}")
    print(f"  Dueluri câșt.:   {int(season_totals.get('duelsWon', 0))}")
    print(f"  Intercepții:     {int(season_totals.get('interceptions', 0))}")

# ── TEST PERCENTILE PE TOȚI JUCĂTORII ────────────────────────
def test_percentile_top10(role_code: str = "MID"):
    print(f"\n{'='*50}")
    print(f"🏆 TOP 10 {role_code} după Percentile Rating")
    print(f"{'='*50}")

    # Citim toți jucătorii cu rolul dat
    players_docs = db.collection("players")\
        .where("role.code3", "==", role_code)\
        .stream()

    player_ids = [doc.to_dict()["wyId"] for doc in players_docs]
    print(f"📋 {len(player_ids)} jucători cu rol {role_code} găsiți")

    if not player_ids:
        print("❌ Niciun jucător găsit pentru acest rol!")
        return

    # Agregăm statisticile sezon pentru fiecare
    all_season_stats = []
    player_names = {}

    for pid in player_ids:
        stats_docs = db.collection("playerStats")\
            .where("playerId", "==", pid)\
            .stream()

        season_totals = defaultdict(float)
        matches_played = 0

        for doc in stats_docs:
            data = doc.to_dict()
            total = data.get("total", {})
            minutes = total.get("minutesOnField", 0) or 0
            if minutes >= 20:
                matches_played += 1
                for key, val in total.items():
                    if isinstance(val, (int, float)):
                        season_totals[key] += val

        if matches_played >= 3:  # minim 3 meciuri pentru rating valid
            all_season_stats.append({
                "playerId": pid,
                "matchesPlayed": matches_played,
                "totalSeason": dict(season_totals),
            })

        # Salvăm numele
        p_doc = db.collection("players").document(str(pid)).get()
        if p_doc.exists:
            player_names[pid] = p_doc.to_dict().get("shortName", str(pid))

    print(f"✅ {len(all_season_stats)} jucători cu minim 3 meciuri")

    if not all_season_stats:
        return

    # Calculăm percentile
    percentiles = calculate_percentile_ratings(all_season_stats, role_code)

    # Sortăm și afișăm top 10
    top10 = sorted(percentiles.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\n{'Loc':<5} {'Jucător':<25} {'Percentile Rating':>18}")
    print("-" * 50)
    for i, (pid, rating) in enumerate(top10, 1):
        name = player_names.get(pid, str(pid))
        print(f"{i:<5} {name:<25} {rating:>18.2f}/10")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: Rating detaliat pentru un jucător specific
    # Înlocuiți cu un wyId real din baza voastră!
    TEST_PLAYER_ID = 1267274
    test_single_player(TEST_PLAYER_ID)

    # Test 2: Top 10 mijlocași după percentile
    test_percentile_top10("MID")