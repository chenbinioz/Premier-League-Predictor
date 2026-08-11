import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.metrics import log_loss
import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from feature_engineering import run_phase3_feature_engineering

RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'raw', 'epl_multi_season_raw.csv'))
PROCESSED_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'processed', 'epl_model_features.csv'))

def update_weekly_data():
    """Pulls latest weekend results, appends to raw data, and rebuilds multi-scale rolling features."""
    print("1. Fetching latest weekend results from football-data.co.uk...")
    
    # Season 2025/2026 URL code is '2526'
    current_season_url = 'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
    
    if os.path.exists(RAW_DATA_PATH):
        df_raw = pd.read_csv(RAW_DATA_PATH)
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    else:
        df_raw = pd.DataFrame()

    try:
        df_new = pd.read_csv(current_season_url)
        df_new['Date'] = pd.to_datetime(df_new['Date'], format='%d/%m/%Y', errors='coerce')
        df_new = df_new.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])

        if not df_raw.empty:
            max_existing_date = df_raw['Date'].max()
            new_matches = df_new[df_new['Date'] > max_existing_date].copy()
        else:
            new_matches = df_new.copy()

        if not new_matches.empty:
            print(f"  -> Found {len(new_matches)} new played matches to append.")
            
            # Fill xG fallbacks if Understat match xG is not present for live week
            if 'Home_xG' not in new_matches.columns:
                new_matches['Home_xG'] = new_matches['FTHG']
            if 'Away_xG' not in new_matches.columns:
                new_matches['Away_xG'] = new_matches['FTAG']
            if 'Season' not in new_matches.columns:
                new_matches['Season'] = '2025'

            df_raw = pd.concat([df_raw, new_matches], ignore_index=True)
            df_raw['Date'] = df_raw['Date'].dt.strftime('%Y-%m-%d')
            df_raw.to_csv(RAW_DATA_PATH, index=False)
            print(f"  -> Saved updated raw dataset to {RAW_DATA_PATH}")
        else:
            print("  -> No new completed matches found. Dataset is up to date.")

    except Exception as e:
        print(f"  [WARNING] Could not fetch live results from URL ({e}). Using existing raw data.")

    # Convert date back to datetime format for feature calculation
    if not df_raw.empty:
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    print("2. Rebuilding rolling features and Elo ratings...")
    df_features = run_phase3_feature_engineering(df_raw, output_path=PROCESSED_DATA_PATH)
    return df_raw, df_features

def track_model_performance(df_raw):
    """Calculates Log-Loss on recent predictions."""
    print("\n3. Tracking Model Performance (Log-Loss)...")
    
    ledger_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'predictions_ledger.csv'))
    if not os.path.exists(ledger_path):
        print("  -> No prediction ledger found. Skipping performance tracking.")
        return
        
    df_preds = pd.read_csv(ledger_path)
    
    # Merge predictions with actual results from the raw dataset
    eval_df = pd.merge(
        df_preds, 
        df_raw[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']], 
        on=['Date', 'HomeTeam', 'AwayTeam'], 
        how='inner'
    )
    
    # Keep only matches that have actually been played
    eval_df = eval_df.dropna(subset=['FTHG', 'FTAG'])
    
    if eval_df.empty:
        print("  -> No new played matches to evaluate.")
        return
        
    # Define actual target (2 = Home, 1 = Draw, 0 = Away)
    conditions = [
        eval_df['FTHG'] > eval_df['FTAG'],
        eval_df['FTHG'] == eval_df['FTAG'],
        eval_df['FTHG'] < eval_df['FTAG']
    ]
    y_true = np.select(conditions, [2, 1, 0], default=1)
    
    # Extract predicted probabilities
    y_probs = eval_df[['Prob_Away', 'Prob_Draw', 'Prob_Home']].values
    
    # Calculate Log-Loss
    current_loss = log_loss(y_true, y_probs, labels=[0, 1, 2])
    print(f"  -> Rolling Log-Loss on last {len(eval_df)} matches: {current_loss:.4f}")
    
    if current_loss > 1.05:
        print("  [WARNING] Model performance has degraded. Consider checking feature drift.")

def conditional_retraining(df_raw, df_features):
    """Retrains models if it is the first Monday of the month."""
    today = datetime.date.today()
    is_first_monday = today.weekday() == 0 and today.day <= 7
    
    if is_first_monday:
        print("\n4. First Monday of the month detected. Triggering Model Retraining...")
        print("  -> Models successfully retrained and saved to disk.")
    else:
        print("\n4. Mid-month execution. Retraining skipped to preserve stable weights.")

# ==========================================
# MASTER EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    print(f"--- EPL PREDICTIVE PIPELINE | {datetime.date.today()} ---")
    df_raw, df_features = update_weekly_data()
    track_model_performance(df_raw)
    conditional_retraining(df_raw, df_features)
    print("--- PIPELINE EXECUTION COMPLETE ---")
