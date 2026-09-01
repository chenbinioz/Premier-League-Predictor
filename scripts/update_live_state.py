import os
import sys
import argparse
import logging
import sqlite3
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# MUST be set before any torch imports to prevent macOS OpenMP issues
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

load_dotenv()

from src.data.state_manager import TeamStateManager
from src.data.live_fetchers import fetch_weekend_results, fetch_live_xg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "").strip()
if not RAPIDAPI_KEY:
    logger.warning(
        "RAPIDAPI_KEY is not configured. The 26/27 tracker will remain on the local seeded data only. "
        "Run this script with a valid API-Football RapidAPI key to refresh results."
    )

DEFAULT_DB_PATH = REPO_ROOT / "data" / "live" / "epl_2627.db"

def auto_discover_gameweeks_to_update(db_path: Path) -> list[int]:
    """Finds all gameweeks that have pending fixtures in the past."""
    if not db_path.exists():
        logger.error(f"Database {db_path} does not exist.")
        return []
    
    today_str = date.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Find unique gameweeks for pending fixtures whose match_date is in the past (<= today)
    cur.execute(
        "SELECT DISTINCT gameweek FROM fixtures_26_27 WHERE status = 'pending' AND match_date <= ?",
        (today_str,)
    )
    rows = cur.fetchall()
    conn.close()
    return [int(r[0]) for r in rows]

def update_gameweek_results(gameweek: int, db_path: Path, dry_run: bool = False):
    logger.info(f"🔄 Processing updates for Gameweek {gameweek}...")
    
    # 1. Fetch results from API-Football
    try:
        results = fetch_weekend_results(gameweek)
    except Exception as e:
        logger.error(f"Failed to fetch results for GW{gameweek}: {e}")
        return
        
    if not results:
        logger.warning(f"No completed results returned from API for GW{gameweek}.")
        return
        
    # Build lookup map: (home_team, away_team) -> result
    results_map = {}
    for r in results:
        key = (r["home_team"], r["away_team"])
        results_map[key] = r
        
    # 2. Open DB and process pending fixtures for this gameweek
    with TeamStateManager(db_path) as sm:
        pending_fixtures = [
            f for f in sm.get_pending_fixtures() if f["gameweek"] == gameweek
        ]
        
        if not pending_fixtures:
            logger.info(f"No pending fixtures in DB for GW{gameweek}.")
            return
            
        updated_count = 0
        for fix in pending_fixtures:
            home = fix["home_team"]
            away = fix["away_team"]
            match_date = fix["match_date"]
            match_id = fix["match_id"]
            
            key = (home, away)
            if key not in results_map:
                logger.debug(f"Fixture {home} vs {away} (GW{gameweek}) not found in API results.")
                continue
                
            res = results_map[key]
            home_goals = res["home_goals"]
            away_goals = res["away_goals"]
            
            # Fetch xG
            logger.info(f"Fetching xG for {home} vs {away} on {match_date}...")
            home_xg, away_xg = fetch_live_xg(home, away, match_date)
            
            if home_xg is None or away_xg is None:
                logger.warning(f"xG fetch failed/returned None. Falling back to goals scored: Home xG={home_goals}, Away xG={away_goals}")
                home_xg = float(home_goals)
                away_xg = float(away_goals)
                
            if dry_run:
                logger.info(f"[DRY RUN] Would update: Match {match_id}: {home} {home_goals} - {away_goals} {away} (xG: {home_xg:.2f} - {away_xg:.2f})")
            else:
                logger.info(f"Applying update: Match {match_id}: {home} {home_goals} - {away_goals} {away} (xG: {home_xg:.2f} - {away_xg:.2f})")
                sm.update_after_match(
                    match_id=match_id,
                    home_team=home,
                    away_team=away,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_xg=home_xg,
                    away_xg=away_xg
                )
                updated_count += 1
                
        logger.info(f"GW{gameweek} processing complete. Updated {updated_count} fixtures.")

def main():
    parser = argparse.ArgumentParser(description="Update live team states with weekend match results.")
    parser.add_argument("--gameweek", "-g", type=int, help="Specify gameweek to update. If omitted, automatically discovers pending past matches.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to SQLite live database.")
    parser.add_argument("--dry-run", action="store_true", help="Print updates without modifying the database.")
    args = parser.parse_args()
    
    db_path = args.db
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
        
    if args.gameweek:
        gameweeks = [args.gameweek]
        logger.info(f"Manually processing gameweek {args.gameweek}.")
    else:
        logger.info("Auto mode: scanning database for pending fixtures in the past...")
        gameweeks = auto_discover_gameweeks_to_update(db_path)
        if not gameweeks:
            logger.info("No pending past fixtures found to update.")
            return
        logger.info(f"Discovered gameweeks needing updates: {gameweeks}")
        
    for gw in gameweeks:
        update_gameweek_results(gw, db_path, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
