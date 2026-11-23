import unittest

import numpy as np
import pandas as pd

from models.neural_dixon_coles import predict_scoreline_grid
from scripts.backtest_challengers import bounded_rho
from scripts.backtest_models import replay


class BacktestReplayTests(unittest.TestCase):
    def test_only_available_results_enter_features(self):
        fixtures = pd.DataFrame([
            {"Date": "2025-08-16", "Time": "16:00", "HomeTeam": "Arsenal",
             "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 0},
            {"Date": "2025-08-17", "Time": "17:00", "HomeTeam": "Arsenal",
             "AwayTeam": "Everton", "FTHG": 0, "FTAG": 1},
        ]).assign(Season="2025/26", HST=6, HS=14, AST=2, AS=9)
        too_early = replay(fixtures, horizon_hours=24, result_lag_hours=3)
        after_result = replay(fixtures, horizon_hours=20, result_lag_hours=3)
        self.assertEqual(too_early.iloc[1].Home_Elo, 1500)
        self.assertGreater(after_result.iloc[1].Home_Elo, 1500)
        self.assertEqual(too_early.iloc[1].Adjusted_Attack_Diff_5, 0)
        self.assertGreater(after_result.iloc[1].Adjusted_Attack_Diff_5, 0)
        self.assertEqual(too_early.iloc[1].Dynamic_Home_Rate, 1.5)
        self.assertGreater(after_result.iloc[1].Dynamic_Home_Rate, 1.5)
        self.assertLess(after_result.iloc[1].Dynamic_Away_Rate, 1.2)
        self.assertEqual(too_early.iloc[1].cutoff_utc, "2025-08-16T16:00:00+00:00")

    def test_missing_kickoff_uses_conservative_bounds(self):
        fixtures = pd.DataFrame([
            {"Date": "2025-08-16", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "FTHG": 2, "FTAG": 0},
            {"Date": "2025-08-18", "Time": "01:00", "HomeTeam": "Arsenal",
             "AwayTeam": "Everton", "FTHG": 0, "FTAG": 1},
        ]).assign(Season="2025/26")
        rows = replay(fixtures, horizon_hours=24, result_lag_hours=3)
        self.assertTrue(rows.iloc[0].kickoff_estimated)
        self.assertEqual(rows.iloc[0].result_available_utc, "2025-08-17T02:00:00+00:00")
        self.assertEqual(rows.iloc[1].Home_Elo, 1500)

    def test_dixon_coles_adjustment_keeps_score_cells_positive(self):
        for strength in (-0.5, 0.5):
            for home, away in ((0.2, 0.4), (7.0, 3.0)):
                with self.subTest(strength=strength, home=home, away=away):
                    rho = bounded_rho(home, away, strength)
                    grid = predict_scoreline_grid(home, away, rho, max_goals=12)
                    self.assertTrue((grid[:2, :2] > 0).all())
                    np.testing.assert_allclose(grid.sum(), 1)
