"""
IO experiment rBergomi pricer (precision-first variant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


XiType = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class MCConfig:
    n_steps_per_year: int = 252
    n_paths: int = 20000
    seed: int = 42
    antithetic: bool = True
    use_control_variate: bool = True
    floor_variance: float = 1e-16


class IOExpRBergomiPricer:
    """Precision-oriented rough Bergomi Monte Carlo pricer."""

    def __init__(self, config: MCConfig = MCConfig()):
        self.config = config

    @staticmethod
    def _option_sign(option_type: str) -> int:
        ot = str(option_type).lower().strip()
        if ot in {"call", "c"}:
            return 1
        if ot in {"put", "p"}:
            return -1
        raise ValueError(f"Unsupported option_type: {option_type}")

    @staticmethod
    def _alpha(H: float) -> float:
        if H <= 0.0 or H >= 0.5:
            raise ValueError("H must be in (0, 0.5).")
        return H - 0.5

    @staticmethod
    def _g(x: float, alpha: float) -> float:
        return x ** alpha

    @staticmethod
    def _b(k: int, alpha: float) -> float:
        numerator = k ** (alpha + 1.0) - (k - 1) ** (alpha + 1.0)
        return (numerator / (alpha + 1.0)) ** (1.0 / alpha)

    @staticmethod
    def _covariance(alpha: float, n_steps_per_year: int) -> np.ndarray:
        n = float(n_steps_per_year)
        cov = np.zeros((2, 2))
        cov[0, 0] = 1.0 / n
        cov[0, 1] = 1.0 / ((alpha + 1.0) * n ** (alpha + 1.0))
        cov[1, 1] = 1.0 / ((2.0 * alpha + 1.0) * n ** (2.0 * alpha + 1.0))
        cov[1, 0] = cov[0, 1]
        return cov

    @staticmethod
    def _build_xi_curve(time_grid: np.ndarray, xi: XiType) -> np.ndarray:
        if np.isscalar(xi):
            xi_curve = np.full_like(time_grid, float(xi), dtype=float)
        elif callable(xi):
            xi_curve = np.asarray(xi(time_grid), dtype=float)
        else:
            xi_curve = np.asarray(xi, dtype=float)
            if xi_curve.shape[0] != time_grid.shape[0]:
                raise ValueError("xi array length must match time grid length.")
        if np.any(xi_curve <= 0):
            raise ValueError("xi(t) must be positive.")
        return xi_curve

    def _make_rng(self, random_seed: Optional[int] = None) -> np.random.Generator:
        seed = self.config.seed if random_seed is None else int(random_seed)
        return np.random.default_rng(seed)

    def _sample_hybrid_increments(
        self,
        n_paths: int,
        n_steps: int,
        H: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        alpha = self._alpha(H)
        cov = self._covariance(alpha, self.config.n_steps_per_year)
        mean = np.zeros(2)
        if not self.config.antithetic:
            return rng.multivariate_normal(mean, cov, size=(n_paths, n_steps))

        n_half = (n_paths + 1) // 2
        base = rng.multivariate_normal(mean, cov, size=(n_half, n_steps))
        anti = -base
        stacked = np.concatenate([base, anti], axis=0)
        return stacked[:n_paths]

    def _sample_std_normal(self, n_paths: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        if not self.config.antithetic:
            return rng.standard_normal(size=(n_paths, n_steps))
        n_half = (n_paths + 1) // 2
        base = rng.standard_normal(size=(n_half, n_steps))
        anti = -base
        stacked = np.concatenate([base, anti], axis=0)
        return stacked[:n_paths]

    def _build_volterra_process(self, dW1: np.ndarray, H: float, n_steps: int) -> np.ndarray:
        alpha = self._alpha(H)
        n_paths = dW1.shape[0]
        y1 = np.zeros((n_paths, n_steps + 1))
        y1[:, 1:] = dW1[:, :, 1]

        g = np.zeros(n_steps + 1)
        for k in range(2, n_steps + 1):
            g[k] = self._g(self._b(k, alpha) / self.config.n_steps_per_year, alpha)

        x = dW1[:, :, 0]
        gx = np.zeros((n_paths, x.shape[1] + g.shape[0] - 1))
        for i in range(n_paths):
            gx[i, :] = np.convolve(g, x[i, :])

        y2 = gx[:, : n_steps + 1]
        scale = np.sqrt(2.0 * alpha + 1.0)
        return scale * (y1 + y2)

    def simulate_paths(
        self,
        S0: float,
        T: float,
        H: float,
        eta: float,
        rho: float,
        xi: XiType = 0.04,
        r: float = 0.0,
        q: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        if T <= 0:
            raise ValueError("T must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if abs(rho) >= 1:
            raise ValueError("rho must be in (-1, 1).")
        if S0 <= 0:
            raise ValueError("S0 must be positive.")

        n_steps = max(1, int(np.ceil(self.config.n_steps_per_year * T)))
        n_paths = int(self.config.n_paths)
        dt = T / n_steps
        time_grid = np.linspace(0.0, T, n_steps + 1)
        rng = self._make_rng(random_seed)

        dW1 = self._sample_hybrid_increments(n_paths, n_steps, H, rng)
        y = self._build_volterra_process(dW1, H, n_steps)

        xi_curve = self._build_xi_curve(time_grid, xi)
        v = xi_curve[None, :] * np.exp(eta * y - 0.5 * (eta ** 2) * (time_grid[None, :] ** (2.0 * H)))

        z = self._sample_std_normal(n_paths, n_steps, rng) * np.sqrt(dt)
        dB = rho * dW1[:, :, 0] + np.sqrt(1.0 - rho ** 2) * z

        drift = (r - q) * dt
        increments = (
            np.sqrt(np.maximum(v[:, :-1], self.config.floor_variance)) * dB
            + drift
            - 0.5 * v[:, :-1] * dt
        )
        log_s = np.zeros((n_paths, n_steps + 1))
        log_s[:, 1:] = np.cumsum(increments, axis=1)
        s = S0 * np.exp(log_s)
        return {"t": time_grid, "S": s, "V": v}

    @staticmethod
    def black_price(
        F: float,
        K: float,
        vol: float,
        T: float,
        option_type: str = "call",
        discount: float = 1.0,
    ) -> float:
        w = IOExpRBergomiPricer._option_sign(option_type)
        if T <= 0:
            return discount * max(w * (F - K), 0.0)
        vol = max(vol, 1e-12)
        std = vol * np.sqrt(T)
        d1 = np.log(F / K) / std + 0.5 * std
        d2 = d1 - std
        return float(discount * (w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)))

    @staticmethod
    def implied_vol_black(
        price: float,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str = "call",
        vol_lower: float = 1e-6,
        vol_upper: float = 5.0,
    ) -> float:
        if T <= 0 or price <= 0:
            return np.nan
        discount = np.exp(-r * T)
        F = S0 * np.exp((r - q) * T)
        w = IOExpRBergomiPricer._option_sign(option_type)
        intrinsic = discount * max(w * (F - K), 0.0)
        target = max(price, intrinsic + 1e-12)

        def err(vol: float) -> float:
            return IOExpRBergomiPricer.black_price(F, K, vol, T, option_type, discount) - target

        try:
            return float(brentq(err, vol_lower, vol_upper))
        except ValueError:
            return np.nan

    def mc_price_from_terminal(
        self,
        s_terminal: np.ndarray,
        K: float,
        T: float,
        r: float,
        q: float,
        S0: float,
        option_type: str,
    ) -> Dict[str, float]:
        w = self._option_sign(option_type)
        discount = np.exp(-r * T)
        payoff = np.maximum(w * (s_terminal - K), 0.0)
        price_raw = float(discount * np.mean(payoff))
        stderr_raw = float(discount * np.std(payoff, ddof=1) / np.sqrt(len(payoff)))

        if not self.config.use_control_variate:
            return {"price": price_raw, "stderr": stderr_raw}

        # Control variate: discounted terminal underlying with known expectation.
        control = discount * s_terminal
        control_mean_theory = S0 * np.exp(-q * T)
        var_c = np.var(control)
        if var_c <= 1e-14:
            return {"price": price_raw, "stderr": stderr_raw}
        cov = np.cov(discount * payoff, control, ddof=1)[0, 1]
        beta = cov / var_c
        adj = discount * payoff - beta * (control - control_mean_theory)
        price_cv = float(np.mean(adj))
        stderr_cv = float(np.std(adj, ddof=1) / np.sqrt(len(adj)))
        return {"price": price_cv, "stderr": stderr_cv}

    def price_cross_section(
        self,
        option_df: pd.DataFrame,
        H: float,
        eta: float,
        rho: float,
        xi: XiType = 0.04,
        random_seed: Optional[int] = None,
    ) -> pd.DataFrame:
        required = {"underlying", "r", "q", "maturity", "strike", "option_type", "market_price"}
        missing = required.difference(option_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = option_df.copy().reset_index(drop=True)
        out["model_price"] = np.nan
        out["model_stderr"] = np.nan
        out["model_iv"] = np.nan
        out["market_iv"] = np.nan

        for i, T in enumerate(np.sort(out["maturity"].unique())):
            sub_idx = out.index[out["maturity"] == T].tolist()
            row0 = out.loc[sub_idx[0]]
            S0 = float(row0["underlying"])
            r = float(row0["r"])
            q = float(row0["q"])

            sim = self.simulate_paths(
                S0=S0,
                T=float(T),
                H=H,
                eta=eta,
                rho=rho,
                xi=xi,
                r=r,
                q=q,
                random_seed=None if random_seed is None else int(random_seed + i),
            )
            s_terminal = sim["S"][:, -1]

            for j in sub_idx:
                row = out.loc[j]
                K = float(row["strike"])
                opt_type = str(row["option_type"])
                market_price = float(row["market_price"])

                pr = self.mc_price_from_terminal(
                    s_terminal=s_terminal,
                    K=K,
                    T=float(T),
                    r=r,
                    q=q,
                    S0=S0,
                    option_type=opt_type,
                )
                model_price = pr["price"]
                model_iv = self.implied_vol_black(model_price, S0, K, float(T), r, q, opt_type)
                market_iv = self.implied_vol_black(market_price, S0, K, float(T), r, q, opt_type)

                out.at[j, "model_price"] = model_price
                out.at[j, "model_stderr"] = pr["stderr"]
                out.at[j, "model_iv"] = model_iv
                out.at[j, "market_iv"] = market_iv

        out["price_error"] = out["model_price"] - out["market_price"]
        out["iv_error"] = out["model_iv"] - out["market_iv"]
        forward = out["underlying"] * np.exp((out["r"] - out["q"]) * out["maturity"])
        out["log_moneyness"] = np.log(out["strike"] / forward)
        return out
