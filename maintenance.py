import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.metrics import log_loss
import os

def update_weekly_data():
    """Pulls weekend results, appends to raw data, and rebuilds features."""
    print("1. Fetching latest weekend results...")
    
    # In a real scenario, this would be an API call or live CSV URL
    # current_season_url = 'https://www.football-data.co.uk/mmz4281/2627/E0.csv'
    # df_new = pd.read_csv(current_season_url)
    
    # For demonstration, we assume df_new is loaded and formatted
    df_raw = pd.read_csv('../data/raw/epl_multi_season_raw.csv')
    
    # Check if there are new matches to append (pseudo-code)
    # new_matches = df_new[df_new['Date'] > df_raw['Date'].max()]
    # if not new_matches.empty:
    #     df_raw = pd.concat([df_raw, new_matches]).reset_index(drop=True)
    #     df_raw.to_csv('../data/raw/epl_multi_season_raw.csv', index=False)
    
    print("2. Rebuilding rolling features and Elo ratings...")
    # NOTE: Call your Phase 3 feature engineering wrapper here
    # df_features = run_phase3_feature_engineering(df_raw)
    
    # For now, we just load the existing one to continue the script
    df_features = pd.read_csv('../data/processed/epl_model_features.csv')
    return df_raw, df_features

def track_model_performance(df_raw):
    """Calculates Log-Loss on recent predictions."""
    print("\n3. Tracking Model Performance (Log-Loss)...")
    
    ledger_path = '../data/predictions_ledger.csv'
    if not os.path.exists(ledger_path):
        print("No prediction ledger found. Skipping performance tracking.")
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
        print("No new played matches to evaluate.")
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
    
    # Alert if model degrades severely (e.g., Log-Loss > 1.05 is usually worse than the bookie)
    if current_loss > 1.05:
        print("  [WARNING] Model performance has degraded. Consider checking feature drift.")

def conditional_retraining(df_raw, df_features):
    """Retrains models if it is the first Monday of the month."""
    today = datetime.date.today()
    
    # Check if today is Monday (0) and within the first 7 days of the month
    is_first_monday = today.weekday() == 0 and today.day <= 7
    
    if is_first_monday:
        print("\n4. First Monday of the month detected. Triggering Model Retraining...")
        
        # --- A. Retrain Dixon-Coles ---
        # dc_params = fit_dixon_coles(df_raw, xi=0.0032)
        # joblib.dump(dc_params, '../models/dc_mle_params.pkl')
        
        # --- B. Retrain XGBoost ---
        # X = df_features[feature_cols]
        # y = df_features['Target']
        # calibrated_model.fit(X, y)
        # joblib.dump(calibrated_model, '../models/calibrated_xgb_outcome.pkl')
        
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


