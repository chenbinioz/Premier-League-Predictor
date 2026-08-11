import streamlit as st
import pandas as pd
import joblib

# 1. Load Data and Models
@st.cache_resource
def load_assets():
    model = joblib.load('models/calibrated_xgb_outcome.pkl')
    features = pd.read_csv('data/processed/epl_model_features.csv')
    
    # Derive differential and engineered features
    features['Elo_Diff'] = features['Home_Elo'] - features['Away_Elo']
    for w in [3, 5, 10]:
        features[f'xG_Attack_Diff_roll{w}'] = features[f'Home_xG_Created_roll{w}'] - features[f'Away_xG_Conceded_roll{w}']
        features[f'xG_Defense_Diff_roll{w}'] = features[f'Away_xG_Created_roll{w}'] - features[f'Home_xG_Conceded_roll{w}']
        features[f'Corner_Diff_roll{w}'] = features[f'Home_Corners_roll{w}'] - features[f'Away_Corners_roll{w}']
        features[f'Foul_Diff_roll{w}'] = features[f'Home_Fouls_roll{w}'] - features[f'Away_Fouls_roll{w}']

    features['Venue_xG_Attack_Diff'] = features['Home_xG_Created_Venue_roll5'] - features['Away_xG_Conceded_Venue_roll5']
    features['Expected_Match_xG'] = features['Home_xG_Created_roll5'] + features['Away_xG_Created_roll5']
    features['Rest_Diff'] = features['Home_Rest_Days'] - features['Away_Rest_Days']
    features['Congestion_Diff'] = features['Home_Congestion_Flag'] - features['Away_Congestion_Flag']

    raw_margin = (1 / features['B365H']) + (1 / features['B365D']) + (1 / features['B365A'])
    features['Bookie_Prob_H'] = (1 / features['B365H']) / raw_margin
    features['Bookie_Prob_D'] = (1 / features['B365D']) / raw_margin
    features['Bookie_Prob_A'] = (1 / features['B365A']) / raw_margin

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