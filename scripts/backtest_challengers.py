"""Experimental form features and score model for historical evaluation."""
from math import exp

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.features.prematch import FeatureEngine, METRICS, mean_history, shot_proxy


ADJUSTED_FEATURES = ["Adjusted_Attack_Diff", "Adjusted_Defense_Diff"]


FORM_WINDOWS = (3, 5, 10)
DYNAMIC_FEATURES = ["Dynamic_Home_Rate", "Dynamic_Away_Rate"]


def form_columns(window):
    return [f"{feature}_{window}" for feature in ADJUSTED_FEATURES]


class ChallengerEngine(FeatureEngine):
    def __init__(self):
        super().__init__()
        self.ratings = {}

    def start_season(self, season, teams):
        if season != self.season:
            self.ratings = {
                team: [0.75 * value for value in self.ratings.get(team, (0.0, 0.0))]
                for team in teams
            }
        super().start_season(season, teams)

    def goal_rates(self, home, away):
        home_attack, home_defense = self.ratings[home]
        away_attack, away_defense = self.ratings[away]
        return 1.5 * exp(home_attack - away_defense), 1.2 * exp(away_attack - home_defense)


    def observe(self, row):
        teams = [self.teams[row[key]] for key in ("home_team", "away_team")]
        metrics = ("xG_Created", "xG_Conceded")
        expectations = [
            [mean_history(team["history"], metric, self.season, 10) for metric in metrics]
            for team in teams
        ]
        quality = [shot_proxy(row.get(f"{side}_sot"), row.get(f"{side}_shots"))
                   for side in ("home", "away")]
        home, away = row["home_team"], row["away_team"]
        home_rate, away_rate = self.goal_rates(home, away)
        super().observe(row)
        # Scale and bound the Poisson residual before moving the log-rate ratings.
        home_error = float(np.clip((row["home_goals"] - home_rate) / home_rate, -2, 2))
        away_error = float(np.clip((row["away_goals"] - away_rate) / away_rate, -2, 2))
        for team, scored, conceded in ((home, home_error, away_error), (away, away_error, home_error)):
            attack, defense = self.ratings[team]
            self.ratings[team] = [0.98 * attack + 0.06 * scored, 0.98 * defense - 0.06 * conceded]

        for team, actual, against, opponent in zip(
            teams, quality, reversed(quality), reversed(expectations)
        ):
            attack, defense = opponent
            team["history"][-1].update(
                adjusted_attack=None if actual is None else actual - (
                    defense + METRICS["xG_Conceded"]) / 2,
                adjusted_defense=None if against is None else against - (
                    attack + METRICS["xG_Created"]) / 2,
            )

    def challenger_features(self, home, away):
        def form(team, metric, window):
            recent = self.teams[team]["history"][-window:]
            # Five matches of neutral prior shrink thin or missing samples.
            known = [(row[metric], .5 ** (int(self.season[:4]) - int(row["season"][:4])))
                     for row in recent if row.get(metric) is not None]
            return sum(value * weight for value, weight in known) / (
                5 + sum(weight for _, weight in known)
            )

        features = dict(zip(DYNAMIC_FEATURES, self.goal_rates(home, away)))
        for window in FORM_WINDOWS:
            for column, metric in zip(form_columns(window), ("adjusted_attack", "adjusted_defense")):
                features[column] = form(home, metric, window) - form(away, metric, window)
        return features


def bounded_rho(home_rate, away_rate, strength):
    """Scale the fitted correction so all four low-score cells stay positive."""
    if strength >= 0:
        limit = np.minimum(1.0, 1 / (home_rate * away_rate))
    else:
        limit = 1 / np.maximum(home_rate, away_rate)
    return strength * limit


def fit_dixon_coles(frame, half_life_days=216):
    teams = sorted(set(frame.HomeTeam) | set(frame.AwayTeam))
    index = {team: i for i, team in enumerate(teams)}
    home, away = frame.HomeTeam.map(index).to_numpy(), frame.AwayTeam.map(index).to_numpy()
    hg, ag = frame.FTHG.to_numpy(), frame.FTAG.to_numpy()
    n = len(teams)
    dates = pd.to_datetime(frame.Date)
    weights = np.exp2(-(dates.max() - dates).dt.days.to_numpy() / half_life_days)

    def unpack(parameters):
        attack = parameters[:n] - parameters[:n].mean()
        return attack, parameters[n:2*n], parameters[-3], parameters[-2], parameters[-1]

    def objective(parameters):
        attack, defense, intercept, advantage, strength = unpack(parameters)
        log_home = intercept + advantage + attack[home] + defense[away]
        log_away = intercept + attack[away] + defense[home]
        lam, mu = np.exp(log_home), np.exp(log_away)
        rho = bounded_rho(lam, mu, strength)
        tau = np.select(
            [(hg == 0) & (ag == 0), (hg == 0) & (ag == 1),
             (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)],
            [1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, 1 - rho],
            default=1.0,
        )
        loss = lam + mu - hg * log_home - ag * log_away - np.log(tau)
        return np.average(loss, weights=weights) + .05 * np.mean(parameters[:2*n] ** 2)

    result = minimize(objective, np.zeros(2*n + 3), method="L-BFGS-B",
                      bounds=[(-2, 2)] * (2*n) + [(-1, 1), (-1, 1), (-.5, .5)])
    if not result.success:
        raise RuntimeError(f"Dixon–Coles fit failed: {result.message}")
    attack, defense, intercept, advantage, strength = unpack(result.x)
    return {"attack": dict(zip(teams, np.exp(attack))), "defense": dict(zip(teams, np.exp(defense))),
            "base_rate": np.exp(intercept), "home_adv": np.exp(advantage), "strength": strength}
