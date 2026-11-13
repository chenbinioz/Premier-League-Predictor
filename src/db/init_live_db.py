"""Create a new live database from an explicit canonical season schedule.

Existing databases are never reset. Use state_manager.rebuild_states for repair.
"""
from contextlib import closing

from pathlib import Path
import argparse
import sqlite3
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.live_fetchers import normalize_team_name,_normalise_date
from src.data.state_manager import TeamStateManager
DB_PATH=ROOT/'data/live/epl_2627.db'


def initialize(db_path,schedule_path):
    db_path=Path(db_path)
    if db_path.exists():
        raise FileExistsError('Database exists; initialization will not overwrite forecasts or results')
    frame=pd.read_csv(schedule_path)
    required={'gameweek','match_date','home_team','away_team'}
    if not required.issubset(frame):
        raise ValueError(f'Schedule must contain {sorted(required)}')
    rows=[]
    for r in frame.to_dict('records'):
        dt=_normalise_date(r['match_date'])
        home,away=normalize_team_name(r['home_team']),normalize_team_name(r['away_team'])
        if not dt or home==away or not home or not away or not 1<=int(r['gameweek'])<=38:
            raise ValueError('Invalid scheduled fixture')
        rows.append((int(r['gameweek']),dt,home,away))
    if len({(r[2],r[3]) for r in rows})!=len(rows):
        raise ValueError('Duplicate scheduled fixture')
    db_path.parent.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executescript((Path(__file__).parent/'schema.sql').read_text())
        conn.executemany('INSERT INTO fixtures_26_27(gameweek,match_date,home_team,away_team) VALUES (?,?,?,?)',rows)
    with TeamStateManager(db_path) as sm:
        with sm._conn:
            sm._conn.execute("UPDATE fixtures_26_27 SET schedule_source='explicit-schedule'")
        if 'schedule_source' in frame and frame.schedule_source.eq('premier-league-fpl').all():
            from src.data.live_fetchers import fetch_official_schedule
            sm.apply_schedule(fetch_official_schedule(schedule_path))
        else:
            sm.rebuild_states()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,default=DB_PATH)
    parser.add_argument('--schedule',type=Path,required=True)
    args=parser.parse_args()
    initialize(args.db,args.schedule)


if __name__=='__main__':main()
