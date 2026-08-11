import pandas as pd
import numpy as np
import os

def calculate_elo(df, k_factor=20, home_advantage=60):
    """Calculates Elo ratings chronologically for all matches."""
    df = df.copy()
    elo_dict = {}
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        if home not in elo_dict:
            elo_dict[home] = 1500
        if away not in elo_dict:
            elo_dict[away] = 1500

        home_elo, away_elo = elo_dict[home], elo_dict[away]
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        prob_home = 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))
        prob_away = 1 - prob_home

        if row['FTHG'] > row['FTAG']:
            res_home, res_away = 1, 0
        elif row['FTHG'] < row['FTAG']:
            res_home, res_away = 0, 1
        else:
            res_home, res_away = 0.5, 0.5

        elo_dict[home] = home_elo + k_factor * (res_home - prob_home)
        elo_dict[away] = away_elo + k_factor * (res_away - prob_away)

    df['Home_Elo'] = home_elos
    df['Away_Elo'] = away_elos
    return df

def run_phase3_feature_engineering(df_raw, output_path='../data/processed/epl_model_features.csv'):
    """
    Restructures raw match data, computes Elo ratings, rest/congestion metrics,
    and multi-scale rolling averages across match windows (3, 5, 10).
    """
    df_raw = df_raw.copy()
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw = df_raw.sort_values('Date').reset_index(drop=True)

    # 1. Calculate Elo Ratings
    df_raw = calculate_elo(df_raw)

    # 2. Restructure into team-centric timeline
    home_cols = {
        'Date': 'Date', 'HomeTeam': 'Team', 'AwayTeam': 'Opponent',
        'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded',
        'Home_xG': 'xG_Created', 'Away_xG': 'xG_Conceded',
        'HY': 'YellowCards', 'HR': 'RedCards', 'HC': 'Corners', 'HF': 'Fouls'
    }
    home_df = df_raw[[col for col in home_cols.keys() if col in df_raw.columns]].rename(columns=home_cols)
    home_df['Venue'] = 'Home'

    away_cols = {
        'Date': 'Date', 'AwayTeam': 'Team', 'HomeTeam': 'Opponent',
        'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded',
        'Away_xG': 'xG_Created', 'Home_xG': 'xG_Conceded',
        'AY': 'YellowCards', 'AR': 'RedCards', 'AC': 'Corners', 'AF': 'Fouls'
    }
    away_df = df_raw[[col for col in away_cols.keys() if col in df_raw.columns]].rename(columns=away_cols)
    away_df['Venue'] = 'Away'

    team_df = pd.concat([home_df, away_df]).sort_values(by=['Team', 'Date']).reset_index(drop=True)

    # 3. Schedule Context
    team_df['Rest_Days'] = team_df.groupby('Team')['Date'].diff().dt.days
    team_df['Rest_Days'] = team_df['Rest_Days'].fillna(14)
    team_df['Congestion_Flag'] = (team_df['Rest_Days'] < 4).astype(int)

    # 4. Multi-Scale Rolling Windows
    windows = [3, 5, 10]
    features_to_roll = [c for c in ['GoalsScored', 'GoalsConceded', 'xG_Created', 'xG_Conceded', 'Corners', 'Fouls'] if c in team_df.columns]

    for w in windows:
        for feat in features_to_roll:
            team_df[f'{feat}_roll{w}'] = team_df.groupby('Team')[feat].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )

    # 5. Venue-Specific Form
    for feat in [c for c in ['xG_Created', 'xG_Conceded'] if c in team_df.columns]:
        team_df[f'{feat}_Venue_roll5'] = team_df.groupby(['Team', 'Venue'])[feat].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )

    # 6. Re-merge back into match format
    home_stats = team_df[team_df['Venue'] == 'Home'].add_prefix('Home_').rename(
        columns={'Home_Date': 'Date', 'Home_Team': 'HomeTeam'}
    )
    away_stats = team_df[team_df['Venue'] == 'Away'].add_prefix('Away_').rename(
        columns={'Away_Date': 'Date', 'Away_Team': 'AwayTeam'}
    )

    base_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Home_Elo', 'Away_Elo']
    if 'Referee' in df_raw.columns:
        base_cols.append('Referee')
    odds_cols = [c for c in df_raw.columns if c in ['B365H', 'B365D', 'B365A']]
    final_df = df_raw[base_cols + odds_cols].copy()

    final_df = pd.merge(final_df, home_stats, on=['Date', 'HomeTeam'], how='inner')
    final_df = pd.merge(final_df, away_stats, on=['Date', 'AwayTeam'], how='inner')
    
    drop_cols = [c for c in ['Home_Opponent', 'Away_Opponent', 'Home_Venue', 'Away_Venue'] if c in final_df.columns]
    final_df = final_df.drop(columns=drop_cols)
    final_df = final_df.dropna().reset_index(drop=True)

    # Save processed feature dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Phase 3 complete! Dataset updated with Elo & rolling features saved to {output_path}.")
    return final_df
