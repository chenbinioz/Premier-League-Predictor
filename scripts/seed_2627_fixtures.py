"""Seed only an explicit schedule; never fabricate fixtures or infer rounds."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.db.init_live_db import main
if __name__=='__main__':main()
