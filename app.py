import streamlit as st
import pandas as pd
import joblib

# 1. Load Data and Models
@st.cache_resource
def load_assets():
    model = joblib.load('models/calibrated_xgb_outcome.pkl')
    features = pd.read_csv('data/processed/epl_model_features.csv')
    return model, features

xgb_model, df_features = load_assets()

# Define the exact feature columns used during model training
feature_cols = [
    'Elo_Diff', 'Home_Elo', 'Away_Elo',
    'xG_Attack_Diff_roll3', 'xG_Defense_Diff_roll3',
    'xG_Attack_Diff_roll5', 'xG_Defense_Diff_roll5',
    'xG_Attack_Diff_roll10', 'xG_Defense_Diff_roll10',
    'Corner_Diff_roll5', 'Foul_Diff_roll5',
    'Venue_xG_Attack_Diff', 'Expected_Match_xG',
    'Rest_Diff', 'Congestion_Diff',
    'Bookie_Prob_H', 'Bookie_Prob_D', 'Bookie_Prob_A'
]

# 2. Build the UI
st.title("Premier League Predictive Engine")
st.markdown("Weekend Forecasts & Win Probabilities")

# 3. User Input
teams = sorted(df_features['HomeTeam'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=0)
with col2:
    away_team = st.selectbox("Away Team", teams, index=1)

# 4. Inference Pipeline
if st.button("Generate Forecast"):
    # Extract the most recent row for this matchup
    match_vector = df_features[
        (df_features['HomeTeam'] == home_team) & 
        (df_features['AwayTeam'] == away_team)
    ].iloc[-1:]
    
    if match_vector.empty:
        st.error(f"No historical data found for {home_team} vs {away_team}.")
    else:
        # Filter down strictly to the numeric feature columns
        X_live = match_vector[feature_cols]
        
        # Execute prediction
        probs = xgb_model.predict_proba(X_live)[0]
        
        st.success(
            f"**{home_team} Win:** {probs[2]:.1%} | "
            f"**Draw:** {probs[1]:.1%} | "
            f"**{away_team} Win:** {probs[0]:.1%}"
        )