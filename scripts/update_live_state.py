"""Reconcile season results and enrichment, then report ingestion freshness."""
from __future__ import annotations
from contextlib import closing
import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.live_fetchers import fetch_results, fetch_official_schedule, fetch_upcoming_odds
from src.data.state_manager import TeamStateManager, utcnow

DEFAULT_DB_PATH = ROOT/'data/live/epl_2627.db'
logger = logging.getLogger(__name__)


def auto_discover_gameweeks_to_update(db_path):
    with closing(sqlite3.connect(db_path)) as conn, conn:
        return [r[0] for r in conn.execute("SELECT DISTINCT gameweek FROM fixtures_26_27 WHERE status='pending' AND schedule_status!='postponed' AND match_date < ? ORDER BY gameweek",(datetime.now(timezone.utc).date().isoformat(),))]


def overdue_fixtures(conn, now=None):
    now = now or datetime.now(timezone.utc)
    return [dict(r) for r in conn.execute("SELECT match_id,home_team,away_team,match_date FROM fixtures_26_27 WHERE status='pending' AND schedule_status!='postponed' AND match_date <= ?",((now-timedelta(days=2)).date().isoformat(),))]


def _reconcile(db_path, results_csv=None, gameweek=None, dry_run=False, include_understat=False, schedule_csv=None, odds_csv=None):
    schedule = fetch_official_schedule(schedule_csv) if not dry_run else None
    try:
        results = fetch_results(results_csv, include_understat=include_understat)
    except RuntimeError:
        if not schedule:
            raise
        results = [dict(home_team=r['home_team'],away_team=r['away_team'],match_date=r['match_date'],
                        home_goals=int(r['official_home_goals']),away_goals=int(r['official_away_goals']),
                        result_source='premier-league-fpl',observed_at=utcnow())
                   for r in schedule if r.get('official_home_goals') is not None and r.get('official_away_goals') is not None]
        if not results:
            raise RuntimeError('Neither results source returned usable completed matches')
    if schedule:
        official={(r['home_team'],r['away_team']):r for r in schedule}
        for result in results:
            reference=official.get((result['home_team'],result['away_team']))
            if reference and reference.get('official_home_goals') is not None:
                if (result['home_goals'],result['away_goals']) != (reference['official_home_goals'],reference['official_away_goals']):
                    raise ValueError(f"Official and CSV scores disagree: {result['home_team']} vs {result['away_team']}")
    quotes = fetch_upcoming_odds(odds_csv) if not dry_run and (odds_csv is not None or results_csv is None) else []
    if dry_run:
        with closing(sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)) as conn, conn:
            conn.row_factory=sqlite3.Row
            known={(r['home_team'],r['away_team']):dict(r) for r in conn.execute('SELECT * FROM fixtures_26_27')}
        candidates=[r for r in results if (r['home_team'],r['away_team']) in known and (gameweek is None or known[(r['home_team'],r['away_team'])]['gameweek']==gameweek)]
        return {'status':'dry-run','source_results':len(results),'matched_fixtures':len(candidates)}
    with TeamStateManager(db_path) as sm:
        sm.apply_schedule(schedule)
        if gameweek is not None:
            keys={(r['home_team'],r['away_team']) for r in sm._conn.execute('SELECT home_team,away_team FROM fixtures_26_27 WHERE gameweek=?',(gameweek,))}
            results=[r for r in results if (r['home_team'],r['away_team']) in keys]
        changed=sm.apply_results(results)
        sm.record_market_quotes(quotes)
        overdue=overdue_fixtures(sm._conn)
        missing_stats=sm._conn.execute("SELECT COUNT(*) FROM fixtures_26_27 WHERE status='completed' AND (home_shots IS NULL OR away_shots IS NULL OR home_shots_on_target IS NULL OR away_shots_on_target IS NULL OR home_corners IS NULL OR away_corners IS NULL OR home_fouls IS NULL OR away_fouls IS NULL)").fetchone()[0]
        summary={'status':'degraded' if overdue or missing_stats else 'ok','source_results':len(results),'changed':changed,'overdue':overdue,
                 'schedule_source':'premier-league-fpl','schedule_fixtures':len(schedule),'missing_match_stats':missing_stats,'observed_quotes':len(quotes),
                 'xg_missing':sm._conn.execute("SELECT COUNT(*) FROM fixtures_26_27 WHERE status='completed' AND (home_xg IS NULL OR away_xg IS NULL)").fetchone()[0]}
        with sm._conn:
            sm._conn.execute('INSERT INTO ingestion_runs(observed_at,status,summary) VALUES (?,?,?)',(utcnow(),summary['status'],json.dumps(summary)))
        if missing_stats:
            raise RuntimeError(f"Results reconciled but {missing_stats} matches lack observed statistics; pipeline is degraded")
        if overdue:
            raise RuntimeError(f'Overdue fixtures unresolved after ingestion: {overdue}')
        return summary


def reconcile(db_path, results_csv=None, gameweek=None, dry_run=False, include_understat=False, schedule_csv=None, odds_csv=None):
    try:
        return _reconcile(db_path,results_csv,gameweek,dry_run,include_understat,schedule_csv,odds_csv)
    except Exception as exc:
        if not dry_run:
            with TeamStateManager(db_path) as sm, sm._conn:
                sm._conn.execute('INSERT INTO ingestion_runs(observed_at,status,summary) VALUES (?,?,?)',(utcnow(),'failed',json.dumps({'error':str(exc)})))
        raise


def update_gameweek_results(gameweek,db_path,dry_run=False):
    return reconcile(db_path,gameweek=gameweek,dry_run=dry_run)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=DEFAULT_DB_PATH)
    parser.add_argument('--results-csv',type=Path)
    parser.add_argument('--schedule-csv',type=Path)
    parser.add_argument('--odds-csv',type=Path)
    parser.add_argument('--understat',action='store_true',help='Attempt optional Understat xG enrichment')
    parser.add_argument('--gameweek','-g',type=int)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--csv-only',action='store_true',help='Skip optional Understat enrichment')
    args=parser.parse_args()
    try:
        print(json.dumps(reconcile(args.db,args.results_csv,args.gameweek,args.dry_run,args.understat and not args.csv_only,args.schedule_csv,args.odds_csv),indent=2))
    except Exception as exc:
        logger.error('Ingestion failed: %s',exc)
        sys.exit(1)
