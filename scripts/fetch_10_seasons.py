"""
Fetch 10 Seasons of EPL Data
============================
Downloads and concatenates 10 seasons of Premier League match data
from Football-Data.co.uk (2016/17 through 2025/26).

Output:
    data/raw/epl_10_seasons_raw.csv
"""

import os
import sys
import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "raw", "epl_10_seasons_raw.csv")

SEASONS = [
    ("1617", "2016/17"),
    ("1718", "2017/18"),
    ("1819", "2018/19"),
    ("1920", "2019/20"),
    ("2021", "2020/21"),
    ("2122", "2021/22"),
    ("2223", "2022/23"),
    ("2324", "2023/24"),
    ("2425", "2024/25"),
    ("2526", "2025/26"),
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"


def fetch_season_data(season_code: str, season_label: str) -> pd.DataFrame:
    url = BASE_URL.format(season_code=season_code)
    print(f"Fetching {season_label} ({url})...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    # Load into dataframe
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
    
    # Filter out empty rows
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    
    # Parse Date with dayfirst=True
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    
    # Assign Season label
    df["Season"] = season_label
    
    print(f"  -> Successfully loaded {len(df)} matches for {season_label}")
    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_dfs = []
    
    for code, label in SEASONS:
        try:
            df_season = fetch_season_data(code, label)
            all_dfs.append(df_season)
        except Exception as e:
            print(f"  [ERROR] Failed to fetch season {label}: {e}")
            sys.exit(1)
            
    # Concatenate all seasons
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort strictly by Date and Time (if present)
    if "Time" in combined_df.columns:
        combined_df["Time_Str"] = combined_df["Time"].fillna("00:00")
        combined_df = combined_df.sort_values(by=["Date", "Time_Str"]).drop(columns=["Time_Str"])
    else:
        combined_df = combined_df.sort_values(by=["Date"])
        
    combined_df = combined_df.reset_index(drop=True)
    
    # Save combined raw data
    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[Success] Fetched {len(combined_df)} matches across {len(SEASONS)} seasons.")
    print(f"Saved raw dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
