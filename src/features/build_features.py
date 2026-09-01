"""
Build Features Module
=====================
Engineers features across 10 seasons of Premier League match data.
Implements:
  1. Team Name Canonicalization across 10 seasons.
  2. Inter-Season Elo Regression to Mean (0.80*Elo + 0.20*1500; promoted = 1420).
  3. Multi-scale rolling form (3, 5, 10 matches) with off-season decay (alpha=0.5).
  4. Schedule context & venue form metrics.
"""

import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "epl_10_seasons_raw.csv")
PROCESSED_DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "epl_model_features.csv")

CANONICAL_TEAMS = {
    "Brighton & Hove Albion": "Brighton",
    "Brighton": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
    "West Bromwich Albion": "West Brom",
    "West Brom": "West Brom",
    "Manchester City": "Man City",
    "Man City": "Man City",
    "Manchester United": "Man United",
    "Man United": "Man United",
    "Newcastle United": "Newcastle",
    "Newcastle": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "Leicester City": "Leicester",
    "Leicester": "Leicester",
    "Leeds United": "Leeds",
    "Leeds": "Leeds",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Luton Town": "Luton",
    "Luton": "Luton",
    "Ipswich Town": "Ipswich",
    "Ipswich": "Ipswich",
    "Cardiff City": "Cardiff",
    "Cardiff": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Huddersfield": "Huddersfield",
    "Middlesbrough": "Middlesbrough",
    "Middlesbrough FC": "Middlesbrough",
    "Stoke City": "Stoke",
    "Stoke": "Stoke",
    "Swansea City": "Swansea",
    "Swansea": "Swansea",
    "Watford": "Watford",
    "Watford FC": "Watford",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",
    "Burnley": "Burnley",
    "Burnley FC": "Burnley",
    "Chelsea": "Chelsea",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Everton FC": "Everton",
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",
    "Hull City": "Hull",
    "Hull": "Hull",
    "Norwich City": "Norwich",
    "Norwich": "Norwich",
    "Southampton": "Southampton",
    "Southampton FC": "Southampton",
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
}


def canonicalize_team_name(name: str) -> str:
    """Map team name variation to canonical string."""
    name_str = str(name).strip()
    return CANONICAL_TEAMS.get(name_str, name_str)


def calculate_10_season_elo(df, k_factor=20, home_advantage=60):
    """
    Calculates Elo ratings chronologically across 10 seasons with inter-season decay.
      - First Season: teams start at 1500.
      - Season Boundary (St -> St+1):
          - Existing teams: Elo_new = 0.80 * Elo_old + 0.20 * 1500
          - Newly promoted teams: Elo_new = 1420
    """
    df = df.copy()
    seasons = sorted(df["Season"].unique().tolist())
    
    elo_dict = {}
    home_elos, away_elos = [], []
    previous_season_teams = set()
    
    for season in seasons:
        df_season = df[df["Season"] == season]
        current_season_teams = set(df_season["HomeTeam"]).union(set(df_season["AwayTeam"]))
        
        if not elo_dict:
            # First season initialization
            for team in current_season_teams:
                elo_dict[team] = 1500.0
        else:
            # Season boundary transition logic
            for team in current_season_teams:
                if team in previous_season_teams:
                    # Existing team: regress to mean
                    elo_dict[team] = 0.80 * elo_dict[team] + 0.20 * 1500.0
                else:
                    # Promoted team: entry rating of 1420
                    elo_dict[team] = 1420.0
                    
        previous_season_teams = current_season_teams
        
        # Process matches in current season
        for idx in df_season.index:
            row = df.loc[idx]
            home, away = row["HomeTeam"], row["AwayTeam"]
            
            home_elo = elo_dict[home]
            away_elo = elo_dict[away]
            
            home_elos.append(home_elo)
            away_elos.append(away_elo)
            
            prob_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_advantage)) / 400.0))
            prob_away = 1.0 - prob_home
            
            if row["FTHG"] > row["FTAG"]:
                res_home, res_away = 1.0, 0.0
            elif row["FTHG"] < row["FTAG"]:
                res_home, res_away = 0.0, 1.0
            else:
                res_home, res_away = 0.5, 0.5
                
            elo_dict[home] = home_elo + k_factor * (res_home - prob_home)
            elo_dict[away] = away_elo + k_factor * (res_away - prob_away)
            
    df["Home_Elo"] = home_elos
    df["Away_Elo"] = away_elos
    return df


def calculate_decay_weighted_rolling_mean(series_vals, season_indices, window, alpha=0.5):
    """
    Computes rolling mean over preceding matches up to `window` with off-season decay.
    Prior season matches carry weight alpha^(current_season_idx - past_season_idx).
    """
    n = len(series_vals)
    result = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        if i == 0:
            result[i] = series_vals[0]
            continue
            
        start_idx = max(0, i - window)
        past_vals = series_vals[start_idx:i]
        past_seasons = season_indices[start_idx:i]
        curr_season = season_indices[i]
        
        weights = alpha ** np.maximum(0, curr_season - past_seasons)
        weighted_mean = np.sum(past_vals * weights) / np.sum(weights)
        result[i] = weighted_mean
        
    return result


def build_features(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    """Reads raw 10-season dataset, computes Elo & decayed rolling form features."""
    print(f"[Build Features] Loading raw dataset from: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    
    # 1. Canonicalize Team Names
    df_raw["HomeTeam"] = df_raw["HomeTeam"].apply(canonicalize_team_name)
    df_raw["AwayTeam"] = df_raw["AwayTeam"].apply(canonicalize_team_name)
    
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    
    # Sort chronologically
    if "Time" in df_raw.columns:
        df_raw["Time_Str"] = df_raw["Time"].fillna("00:00")
        df_raw = df_raw.sort_values(by=["Date", "Time_Str"]).drop(columns=["Time_Str"])
    else:
        df_raw = df_raw.sort_values(by=["Date"])
    df_raw = df_raw.reset_index(drop=True)
    
    # 2. Calculate 10-Season Elo Ratings with Inter-Season Decay
    print("[Build Features] Computing 10-season Elo ratings with inter-season decay...")
    df_raw = calculate_10_season_elo(df_raw)
    
    # 3. Estimate xG if not directly present
    if "Home_xG" not in df_raw.columns and "HST" in df_raw.columns:
        df_raw["Home_xG"] = 0.32 * df_raw["HST"] + 0.03 * (df_raw["HS"] - df_raw["HST"])
    elif "Home_xG" not in df_raw.columns:
        df_raw["Home_xG"] = df_raw["FTHG"]
        
    if "Away_xG" not in df_raw.columns and "AST" in df_raw.columns:
        df_raw["Away_xG"] = 0.32 * df_raw["AST"] + 0.03 * (df_raw["AS"] - df_raw["AST"])
    elif "Away_xG" not in df_raw.columns:
        df_raw["Away_xG"] = df_raw["FTAG"]
        
    # 4. Restructure into team-centric timeline
    home_cols = {
        "Date": "Date", "Season": "Season", "HomeTeam": "Team", "AwayTeam": "Opponent",
        "FTHG": "GoalsScored", "FTAG": "GoalsConceded",
        "Home_xG": "xG_Created", "Away_xG": "xG_Conceded",
        "HST": "ShotsOnTarget", "AST": "Opp_ShotsOnTarget",
        "HY": "YellowCards", "HR": "RedCards", "HC": "Corners", "HF": "Fouls"
    }
    home_df = df_raw[[c for c in home_cols.keys() if c in df_raw.columns]].rename(columns=home_cols)
    home_df["Venue"] = "Home"
    
    away_cols = {
        "Date": "Date", "Season": "Season", "AwayTeam": "Team", "HomeTeam": "Opponent",
        "FTAG": "GoalsScored", "FTHG": "GoalsConceded",
        "Away_xG": "xG_Created", "Home_xG": "xG_Conceded",
        "AST": "ShotsOnTarget", "HST": "Opp_ShotsOnTarget",
        "AY": "YellowCards", "AR": "RedCards", "AC": "Corners", "AF": "Fouls"
    }
    away_df = df_raw[[c for c in away_cols.keys() if c in df_raw.columns]].rename(columns=away_cols)
    away_df["Venue"] = "Away"
    
    team_df = pd.concat([home_df, away_df]).sort_values(by=["Team", "Date"]).reset_index(drop=True)
    
    # Season numeric index for decay calculation
    season_labels = sorted(team_df["Season"].unique().tolist())
    season_to_idx = {s: i for i, s in enumerate(season_labels)}
    team_df["Season_Idx"] = team_df["Season"].map(season_to_idx)
    
    # 5. Schedule Context
    team_df["Rest_Days"] = team_df.groupby("Team")["Date"].diff().dt.days
    team_df["Rest_Days"] = team_df["Rest_Days"].fillna(14)
    team_df["Congestion_Flag"] = (team_df["Rest_Days"] < 4).astype(int)
    
    # 6. Decayed Rolling Features
    windows = [3, 5, 10]
    features_to_roll = [c for c in ["GoalsScored", "GoalsConceded", "ShotsOnTarget", "xG_Created", "xG_Conceded", "Corners", "Fouls"] if c in team_df.columns]
    
    print("[Build Features] Computing decayed rolling metrics (windows 3, 5, 10, alpha=0.5)...")
    for w in windows:
        for feat in features_to_roll:
            col_name = f"{feat}_roll{w}"
            team_df[col_name] = 0.0
            
            for team, group in team_df.groupby("Team"):
                vals = group[feat].values
                s_idx = group["Season_Idx"].values
                rolled_vals = calculate_decay_weighted_rolling_mean(vals, s_idx, window=w, alpha=0.5)
                team_df.loc[group.index, col_name] = rolled_vals
                
    # 7. Venue-Specific Form
    for feat in [c for c in ["xG_Created", "xG_Conceded"] if c in team_df.columns]:
        col_name = f"{feat}_Venue_roll5"
        team_df[col_name] = 0.0
        
        for (team, venue), group in team_df.groupby(["Team", "Venue"]):
            vals = group[feat].values
            s_idx = group["Season_Idx"].values
            rolled_vals = calculate_decay_weighted_rolling_mean(vals, s_idx, window=5, alpha=0.5)
            team_df.loc[group.index, col_name] = rolled_vals
            
    # 8. Re-merge back into match format
    home_stats = team_df[team_df["Venue"] == "Home"].add_prefix("Home_").rename(
        columns={"Home_Date": "Date", "Home_Team": "HomeTeam"}
    )
    away_stats = team_df[team_df["Venue"] == "Away"].add_prefix("Away_").rename(
        columns={"Away_Date": "Date", "Away_Team": "AwayTeam"}
    )
    
    base_cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Home_Elo", "Away_Elo"]
    if "Referee" in df_raw.columns:
        base_cols.append("Referee")
    odds_cols = [c for c in df_raw.columns if c in ["B365H", "B365D", "B365A"]]
    final_df = df_raw[base_cols + odds_cols].copy()
    
    final_df = pd.merge(final_df, home_stats, on=["Date", "HomeTeam"], how="inner")
    final_df = pd.merge(final_df, away_stats, on=["Date", "AwayTeam"], how="inner")
    
    drop_cols = [c for c in ["Home_Opponent", "Away_Opponent", "Home_Venue", "Away_Venue", "Home_Season", "Away_Season", "Home_Season_Idx", "Away_Season_Idx"] if c in final_df.columns]
    final_df = final_df.drop(columns=drop_cols)
    final_df = final_df.dropna().reset_index(drop=True)
    
    # Save processed feature dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[Build Features] ✅ Complete! Processed dataset with {len(final_df)} rows and {len(final_df.columns)} columns saved to {output_path}.")
    return final_df


if __name__ == "__main__":
    build_features()
