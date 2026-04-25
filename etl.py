import json
import os
import glob
import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict

# Setup Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

PLAYERS_JSON = "players.json"
STATS_FOLDER = "brute_data/"

# ── 1. Load players ──────────────────────────────────────────
def load_players():
    with open(PLAYERS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    
    players = data["players"]
    batch = db.batch()
    count = 0
    
    for p in players:
        doc_ref = db.collection("players").document(str(p["wyId"]))
        batch.set(doc_ref, {
            "wyId": p["wyId"],
            "shortName": p.get("shortName", ""),
            "firstName": p.get("firstName", ""),
            "lastName": p.get("lastName", "").strip(),
            "birthDate": p.get("birthDate"),
            "height": p.get("height", 0),
            "weight": p.get("weight", 0),
            "foot": p.get("foot"),
            "role": p.get("role", {}),
            "birthArea": p.get("birthArea", {}),
            "currentTeamId": p.get("currentTeamId"),
            "status": p.get("status", "active"),
        }, merge=True)
        count += 1
        
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()
    
    batch.commit()
    print(f"✅ Loaded {count} players")

# ── 2. Build teams ────────────────────────────────────────────
def build_teams():
    with open(PLAYERS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    
    teams = defaultdict(list)
    for p in data["players"]:
        tid = p.get("currentTeamId")
        if tid:
            teams[tid].append(p["wyId"])
    
    batch = db.batch()
    for team_id, player_ids in teams.items():
        doc_ref = db.collection("teams").document(str(team_id))
        batch.set(doc_ref, {
            "teamId": team_id,
            "playerIds": player_ids,
            "name": f"Team_{team_id}",
        }, merge=True)
    
    batch.commit()
    print(f"✅ Built {len(teams)} teams")

# ── 3. Load match stats ───────────────────────────────────────
def load_match_stats():
    files = glob.glob(os.path.join(STATS_FOLDER, "*_players_stats.json"))
    total_docs = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        match_label = filename.replace("_players_stats.json", "")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        players_stats = data if isinstance(data, list) else data.get("players", [])

        if not players_stats:
            print(f"  ⚠️  {filename}: gol sau format necunoscut")
            continue

        batch = db.batch()
        count = 0
        match_id = None

        for stat in players_stats:
            player_id = stat.get("playerId") or stat.get("wyId")
            match_id = stat.get("matchId")

            if not player_id or not match_id:
                continue

            doc_id = f"{player_id}_{match_id}"
            doc_ref = db.collection("playerStats").document(doc_id)

            batch.set(doc_ref, {
                "playerId": player_id,
                "matchId": match_id,
                "matchLabel": match_label,
                "competitionId": stat.get("competitionId"),
                "seasonId": stat.get("seasonId"),
                "roundId": stat.get("roundId"),
                "positions": stat.get("positions", []),
                "total": stat.get("total", {}),
                "average": stat.get("average", {}),
                "percent": stat.get("percent", {}),
            })

            count += 1
            total_docs += 1

            if count % 400 == 0:
                batch.commit()
                batch = db.batch()

        batch.commit()

        # Salvează metadata meci
        if match_id:
            try:
                teams_part = match_label.split(",")[0]   # "Argeș - Botoșani"
                score_part = match_label.split(",")[1].strip()  # "0-0"
                home_team = teams_part.split(" - ")[0].strip()
                away_team = teams_part.split(" - ")[1].strip()
            except:
                home_team = away_team = score_part = ""

            db.collection("matches").document(str(match_id)).set({
                "matchId": match_id,
                "label": match_label,
                "homeTeam": home_team,
                "awayTeam": away_team,
                "score": score_part,
                "competitionId": players_stats[0].get("competitionId"),
                "seasonId": players_stats[0].get("seasonId"),
                "roundId": players_stats[0].get("roundId"),
            }, merge=True)

        print(f"  📁 {match_label}: {count} stats")

    print(f"✅ Total stats loaded: {total_docs}")

# ── 4. Infer team names din match labels ──────────────────────
def infer_team_names():
    """
    Din label-ul meciului "Argeș - Botoșani, 0-0" extragem numele echipelor
    și le asociem cu teamId-urile din players.json prin currentTeamId.
    """
    files = glob.glob(os.path.join(STATS_FOLDER, "*_players_stats.json"))
    team_name_map = {}  # teamId -> set of possible names

    for file_path in files:
        filename = os.path.basename(file_path)
        match_label = filename.replace("_players_stats.json", "")

        try:
            teams_part = match_label.split(",")[0]
            home_name = teams_part.split(" - ")[0].strip()
            away_name = teams_part.split(" - ")[1].strip()
        except:
            continue

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        players_stats = data if isinstance(data, list) else data.get("players", [])
        if not players_stats:
            continue

        # Colectăm player IDs din meci
        player_ids = [
            str(s.get("playerId") or s.get("wyId"))
            for s in players_stats
            if s.get("playerId") or s.get("wyId")
        ]

        # Lookup currentTeamId din Firestore pentru primii 30 de jucători
        team_ids_found = defaultdict(int)
        for pid in player_ids[:30]:
            doc = db.collection("players").document(pid).get()
            if doc.exists:
                tid = doc.to_dict().get("currentTeamId")
                if tid:
                    team_ids_found[tid] += 1

        # Sortăm după frecvență — primele 2 sunt cele 2 echipe
        sorted_teams = sorted(team_ids_found.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_teams) >= 2:
            team_name_map[sorted_teams[0][0]] = home_name
            team_name_map[sorted_teams[1][0]] = away_name
        elif len(sorted_teams) == 1:
            team_name_map[sorted_teams[0][0]] = home_name

    # Update teams în Firestore
    batch = db.batch()
    count = 0
    for team_id, name in team_name_map.items():
        doc_ref = db.collection("teams").document(str(team_id))
        batch.set(doc_ref, {"name": name}, merge=True)
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()

    print(f"✅ Inferred names for {len(team_name_map)} teams")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting ETL...")
    load_players()
    build_teams()
    load_match_stats()
    print("🔍 Inferring team names (poate dura 1-2 min)...")
    infer_team_names()
    print("🎉 ETL complet! Verificați Firestore.")