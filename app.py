import streamlit as st
import pandas as pd
import joblib

# 1. Load Data and Models
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_assets():
    model = joblib.load('models/calibrated_xgb_outcome.pkl')
    features = pd.read_csv('data/processed/epl_model_features.csv')
    return model, features

xgb_model, df_features = load_assets()

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

# 4. Inference (Simplified)
if st.button("Generate Forecast"):
    # Extract the most recent feature vector for this matchup
    match_vector = df_features[
        (df_features['HomeTeam'] == home_team) & 
        (df_features['AwayTeam'] == away_team)
    ].iloc[-1:]
    
    # Predict
    probs = xgb_model.predict_proba(match_vector)[0]
    
    st.success(f"{home_team} Win: {probs[2]:.1%} | Draw: {probs[1]:.1%} | {away_team} Win: {probs[0]:.1%}")