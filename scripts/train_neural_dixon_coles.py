"""Compatibility entry point for the versioned, pre-match-only bundle trainer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_models import train
if __name__=='__main__':train()
