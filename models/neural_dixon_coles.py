"""
Neural Dixon-Coles Model
========================
A PyTorch model that maps engineered match features to match-specific expected
goals (lambda, mu) and low-score dependency correction (rho).

Components:
- NeuralDixonColes: nn.Module with two-layer backbone and three output heads.
- DixonColesLoss: Custom NLL combining the tau correction with Poisson PMF.
- predict_scoreline_grid: Produces a normalised 7x7 probability matrix.
- parse_grid_outputs: Extracts 1X2 probabilities and top-K scorelines.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import poisson as scipy_poisson


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class NeuralDixonColes(nn.Module):
    """
    Two-layer backbone with three output heads:
      - head_lambda  → Softplus  → λ > 0 (Home expected goals)
      - head_mu      → Softplus  → μ > 0 (Away expected goals)
      - head_rho     → Tanh×0.25 → ρ ∈ (-0.25, 0.25) (Low-score dependency)
    """

    def __init__(self, n_features: int):
        super().__init__()

        # --- Backbone ---
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )

        # --- Output Heads ---
        self.head_lambda = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.head_mu     = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        # Tanh scaled to (-0.25, 0.25)
        self.head_rho_fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        lam = self.head_lambda(h).squeeze(-1)      # (batch,)
        mu  = self.head_mu(h).squeeze(-1)           # (batch,)
        rho = torch.tanh(self.head_rho_fc(h)).squeeze(-1) * 0.25  # (batch,)
        return lam, mu, rho


# ---------------------------------------------------------------------------
# Custom NLL Loss
# ---------------------------------------------------------------------------

class DixonColesLoss(nn.Module):
    """
    Negative Log-Likelihood for the Dixon-Coles model.

    L = -mean[ ln τ(λ,μ,ρ,yH,yA) + ln Poisson(yH;λ) + ln Poisson(yA;μ) ]

    The τ (tau) low-score correction is:
        τ(λ,μ,ρ,0,0) = 1 - λ·μ·ρ
        τ(λ,μ,ρ,1,0) = 1 + μ·ρ
        τ(λ,μ,ρ,0,1) = 1 + λ·ρ
        τ(λ,μ,ρ,1,1) = 1 - ρ
        τ(λ,μ,ρ,h,a) = 1  for h+a > 2
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def _tau(
        self,
        lam: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
        yh: torch.Tensor,
        ya: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample tau correction factor."""
        tau = torch.ones_like(lam)

        mask_00 = (yh == 0) & (ya == 0)
        mask_10 = (yh == 1) & (ya == 0)
        mask_01 = (yh == 0) & (ya == 1)
        mask_11 = (yh == 1) & (ya == 1)

        tau = torch.where(mask_00, 1.0 - lam * mu * rho, tau)
        tau = torch.where(mask_10, 1.0 + mu * rho, tau)
        tau = torch.where(mask_01, 1.0 + lam * rho, tau)
        tau = torch.where(mask_11, 1.0 - rho, tau)
        return tau

    def _log_poisson_pmf(
        self, k: torch.Tensor, rate: torch.Tensor
    ) -> torch.Tensor:
        """Numerically stable log Poisson PMF: k·log(rate) - rate - log(k!)"""
        # Use torch's built-in lgamma for log factorial: log(k!) = lgamma(k+1)
        log_pmf = k * torch.log(rate.clamp(min=self.eps)) - rate - torch.lgamma(k + 1.0)
        return log_pmf

    def forward(
        self,
        lam: torch.Tensor,
        mu: torch.Tensor,
        rho: torch.Tensor,
        yh: torch.Tensor,
        ya: torch.Tensor,
    ) -> torch.Tensor:
        yh_f = yh.float()
        ya_f = ya.float()

        tau    = self._tau(lam, mu, rho, yh_f, ya_f)
        log_tau = torch.log(tau.clamp(min=self.eps))

        log_ph = self._log_poisson_pmf(yh_f, lam)
        log_pa = self._log_poisson_pmf(ya_f, mu)

        nll = -(log_tau + log_ph + log_pa).mean()
        return nll


# ---------------------------------------------------------------------------
# Scoreline Grid
# ---------------------------------------------------------------------------

def predict_scoreline_grid(
    lambda_h: float,
    mu_a: float,
    rho: float,
    max_goals: int = 6,
) -> np.ndarray:
    """
    Return a normalised (max_goals+1) × (max_goals+1) probability matrix.

    Rows index home goals (0..max_goals), columns index away goals (0..max_goals).
    Dixon-Coles low-score correction (tau) is applied to the (0,0),(1,0),(0,1),(1,1) cells.

    Parameters
    ----------
    lambda_h : float  – Predicted home expected goals.
    mu_a     : float  – Predicted away expected goals.
    rho      : float  – Low-score dependency parameter.
    max_goals: int    – Maximum goals per team in the grid (default 6 → 7×7 grid).

    Returns
    -------
    grid : np.ndarray, shape (max_goals+1, max_goals+1), sums to ~1.0
    """
    n = max_goals + 1
    home_pmf = scipy_poisson.pmf(np.arange(n), lambda_h)
    away_pmf = scipy_poisson.pmf(np.arange(n), mu_a)
    grid = np.outer(home_pmf, away_pmf)

    # Apply Dixon-Coles tau corrections
    eps = 1e-10

    def _tau(h: int, a: int) -> float:
        if h == 0 and a == 0:
            return max(1.0 - lambda_h * mu_a * rho, eps)
        elif h == 1 and a == 0:
            return 1.0 + mu_a * rho
        elif h == 0 and a == 1:
            return 1.0 + lambda_h * rho
        elif h == 1 and a == 1:
            return max(1.0 - rho, eps)
        else:
            return 1.0

    for h in range(min(2, n)):
        for a in range(min(2, n)):
            grid[h, a] *= _tau(h, a)

    # Normalise so grid sums to 1
    total = grid.sum()
    if total > 0:
        grid /= total

    return grid


def parse_grid_outputs(
    grid: np.ndarray,
    top_k: int = 5,
) -> dict:
    """
    Derive 1X2 probabilities and top-K exact scorelines from a scoreline grid.

    Parameters
    ----------
    grid  : np.ndarray – (max_goals+1) × (max_goals+1) normalised probability matrix.
    top_k : int        – Number of top scorelines to return (default 5).

    Returns
    -------
    dict with keys:
        'home_win'    : float (probability as %)
        'draw'        : float (probability as %)
        'away_win'    : float (probability as %)
        'top_scorelines': list of dicts [{'scoreline': str, 'probability': str}, ...]
    """
    n = grid.shape[0]

    home_win = float(np.sum(np.tril(grid, k=-1)))  # rows > cols
    draw     = float(np.sum(np.diag(grid)))          # diagonal
    away_win = float(np.sum(np.triu(grid, k=1)))    # cols > rows

    # Normalise 1X2 to 100%
    total_1x2 = home_win + draw + away_win
    if total_1x2 > 0:
        home_win /= total_1x2
        draw     /= total_1x2
        away_win /= total_1x2

    # Top-K scorelines
    flat_idx  = np.argsort(grid.ravel())[::-1][:top_k]
    rows, cols = np.unravel_index(flat_idx, grid.shape)
    top_scorelines = [
        {
            "scoreline":   f"{r} - {c}",
            "probability": f"{grid[r, c]:.1%}",
        }
        for r, c in zip(rows, cols)
    ]

    return {
        "home_win":       round(home_win * 100, 2),
        "draw":           round(draw * 100, 2),
        "away_win":       round(away_win * 100, 2),
        "top_scorelines": top_scorelines,
    }
