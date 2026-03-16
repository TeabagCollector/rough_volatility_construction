"""
rBergomi pricer.

Implements rough Bergomi path simulation, Monte Carlo pricing for European options,
and Black implied-vol inversion for smile/term-structure comparison.
"""

from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


class rBergomiPricer:
    """Monte Carlo pricer for the rough Bergomi model."""

    def __init__(
        self,
        n_steps_per_year: int = 252,
        n_paths: int = 5000,
        seed: Optional[int] = 42
    ):
        self.n_steps_per_year = int(n_steps_per_year)
        self.n_paths = int(n_paths)
        self.seed = seed

    @staticmethod
    def _alpha(H: float) -> float:
        """Alpha parameter in the Volterra kernel: alpha = H - 1/2."""
        if H <= 0.0 or H >= 0.5:
            raise ValueError("H must be in (0, 0.5) for rough volatility.")
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
        """2x2 covariance of the hybrid scheme increments."""
        n = float(n_steps_per_year)
        cov = np.zeros((2, 2))
        cov[0, 0] = 1.0 / n
        cov[0, 1] = 1.0 / ((alpha + 1.0) * n ** (alpha + 1.0))
        cov[1, 1] = 1.0 / ((2.0 * alpha + 1.0) * n ** (2.0 * alpha + 1.0))
        cov[1, 0] = cov[0, 1]
        return cov

    @staticmethod
    def _option_sign(option_type: str) -> int:
        ot = str(option_type).lower().strip()
        if ot in {"call", "c"}:
            return 1
        if ot in {"put", "p"}:
            return -1
        raise ValueError(f"Unsupported option_type: {option_type}")

    def _make_rng(self, random_seed: Optional[int] = None) -> np.random.Generator:
        seed = self.seed if random_seed is None else random_seed
        return np.random.default_rng(seed)

    @staticmethod
    def _build_xi_curve(
        time_grid: np.ndarray,
        xi: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]
    ) -> np.ndarray:
        """
        Build forward variance curve xi_0(t) on the simulation grid.
        """
        if np.isscalar(xi):
            xi_curve = np.full_like(time_grid, float(xi), dtype=float)
        elif callable(xi):
            xi_curve = np.asarray(xi(time_grid), dtype=float)
        else:
            xi_curve = np.asarray(xi, dtype=float)
            if xi_curve.shape[0] != time_grid.shape[0]:
                raise ValueError("xi array length must match time grid length.")

        if np.any(xi_curve <= 0):
            raise ValueError("xi(t) must be strictly positive.")
        return xi_curve

    def _sample_hybrid_increments(
        self,
        n_paths: int,
        n_steps: int,
        H: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample 2D Gaussian increments for the hybrid scheme.
        Shape: (n_paths, n_steps, 2).
        """
        alpha = self._alpha(H)
        cov = self._covariance(alpha, self.n_steps_per_year)
        mean = np.zeros(2)
        return rng.multivariate_normal(mean, cov, size=(n_paths, n_steps))

    def _build_volterra_process(
        self,
        dW1: np.ndarray,
        H: float,
        n_steps: int
    ) -> np.ndarray:
        """
        Construct Y_t (fractional Gaussian Volterra process) with hybrid scheme.
        Returns shape (n_paths, n_steps + 1).
        """
        alpha = self._alpha(H)
        n_paths = dW1.shape[0]

        Y1 = np.zeros((n_paths, n_steps + 1))
        Y1[:, 1:] = dW1[:, :, 1]

        G = np.zeros(n_steps + 1)
        for k in range(2, n_steps + 1):
            G[k] = self._g(self._b(k, alpha) / self.n_steps_per_year, alpha)

        X = dW1[:, :, 0]
        GX = np.zeros((n_paths, X.shape[1] + G.shape[0] - 1))
        for i in range(n_paths):
            GX[i, :] = np.convolve(G, X[i, :])

        Y2 = GX[:, : n_steps + 1]
        scale = np.sqrt(2.0 * alpha + 1.0)
        return scale * (Y1 + Y2)

    def simulate_paths(
        self,
        S0: float,
        T: float,
        H: float,
        eta: float,
        rho: float,
        xi: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0.04,
        r: float = 0.0,
        q: float = 0.0,
        random_seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate rBergomi variance and stock paths on [0, T].
        """
        if T <= 0:
            raise ValueError("T must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if abs(rho) >= 1:
            raise ValueError("rho must be in (-1, 1).")
        if S0 <= 0:
            raise ValueError("S0 must be positive.")

        n_steps = max(1, int(np.ceil(self.n_steps_per_year * T)))
        dt = T / n_steps
        time_grid = np.linspace(0.0, T, n_steps + 1)
        rng = self._make_rng(random_seed)

        dW1 = self._sample_hybrid_increments(self.n_paths, n_steps, H, rng)
        Y = self._build_volterra_process(dW1, H, n_steps)

        xi_curve = self._build_xi_curve(time_grid, xi)
        V = xi_curve[None, :] * np.exp(eta * Y - 0.5 * (eta ** 2) * (time_grid[None, :] ** (2.0 * H)))

        dW2 = rng.standard_normal(size=(self.n_paths, n_steps)) * np.sqrt(dt)
        dB = rho * dW1[:, :, 0] + np.sqrt(1.0 - rho ** 2) * dW2

        drift = (r - q) * dt
        increments = np.sqrt(np.maximum(V[:, :-1], 1e-16)) * dB + drift - 0.5 * V[:, :-1] * dt
        log_S = np.zeros((self.n_paths, n_steps + 1))
        log_S[:, 1:] = np.cumsum(increments, axis=1)
        S = S0 * np.exp(log_S)

        return {
            "t": time_grid,
            "V": V,
            "S": S
        }

    @staticmethod
    def mc_price_from_terminal(
        S_terminal: np.ndarray,
        K: float,
        T: float,
        r: float = 0.0,
        option_type: str = "call"
    ) -> float:
        """Discounted Monte Carlo price from terminal asset values."""
        w = rBergomiPricer._option_sign(option_type)
        payoff = np.maximum(w * (S_terminal - K), 0.0)
        return float(np.exp(-r * T) * np.mean(payoff))

    @staticmethod
    def black_price(
        F: float,
        K: float,
        vol: float,
        T: float,
        option_type: str = "call",
        discount: float = 1.0
    ) -> float:
        """Black-76 option price on forward F."""
        if T <= 0:
            intrinsic = max(rBergomiPricer._option_sign(option_type) * (F - K), 0.0)
            return discount * intrinsic

        vol = max(vol, 1e-12)
        std = vol * np.sqrt(T)
        w = rBergomiPricer._option_sign(option_type)
        d1 = np.log(F / K) / std + 0.5 * std
        d2 = d1 - std
        return float(discount * (w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)))

    @staticmethod
    def implied_vol_black(
        price: float,
        S0: float,
        K: float,
        T: float,
        r: float = 0.0,
        q: float = 0.0,
        option_type: str = "call",
        vol_lower: float = 1e-6,
        vol_upper: float = 5.0
    ) -> float:
        """Black implied volatility from option price."""
        if T <= 0 or price <= 0:
            return np.nan

        discount = np.exp(-r * T)
        F = S0 * np.exp((r - q) * T)
        w = rBergomiPricer._option_sign(option_type)
        intrinsic = discount * max(w * (F - K), 0.0)
        target = max(price, intrinsic + 1e-12)

        def err(vol: float) -> float:
            model_price = rBergomiPricer.black_price(
                F=F, K=K, vol=vol, T=T, option_type=option_type, discount=discount
            )
            return model_price - target

        try:
            return float(brentq(err, vol_lower, vol_upper))
        except ValueError:
            return np.nan

    def price_single_option(
        self,
        S0: float,
        K: float,
        T: float,
        H: float,
        eta: float,
        rho: float,
        xi: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0.04,
        r: float = 0.0,
        q: float = 0.0,
        option_type: str = "call",
        random_seed: Optional[int] = None
    ) -> Dict[str, float]:
        """Price one option under rBergomi and report model IV."""
        sim = self.simulate_paths(
            S0=S0, T=T, H=H, eta=eta, rho=rho, xi=xi, r=r, q=q, random_seed=random_seed
        )
        price = self.mc_price_from_terminal(
            S_terminal=sim["S"][:, -1], K=K, T=T, r=r, option_type=option_type
        )
        iv = self.implied_vol_black(
            price=price, S0=S0, K=K, T=T, r=r, q=q, option_type=option_type
        )
        return {"price": price, "iv": iv}

    def price_cross_section(
        self,
        option_df: pd.DataFrame,
        H: float,
        eta: float,
        rho: float,
        xi: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0.04,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Price an option cross-section DataFrame.
        Required columns:
        S0, r, q, maturity, strike, option_type, market_price
        """
        required = {"S0", "r", "q", "maturity", "strike", "option_type", "market_price"}
        missing = required.difference(option_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = option_df.copy().reset_index(drop=True)
        model_prices = []
        model_ivs = []
        market_ivs = []

        for i, row in out.iterrows():
            S0 = float(row["S0"])
            r = float(row["r"])
            q = float(row["q"])
            T = float(row["maturity"])
            K = float(row["strike"])
            option_type = str(row["option_type"])
            market_price = float(row["market_price"])

            # Deterministic objective: fixed per-row seed offset for calibration stability.
            row_seed = None if random_seed is None else int(random_seed + i)
            result = self.price_single_option(
                S0=S0, K=K, T=T, H=H, eta=eta, rho=rho, xi=xi, r=r, q=q,
                option_type=option_type, random_seed=row_seed
            )

            model_price = result["price"]
            model_iv = result["iv"]
            market_iv = self.implied_vol_black(
                price=market_price, S0=S0, K=K, T=T, r=r, q=q, option_type=option_type
            )

            model_prices.append(model_price)
            model_ivs.append(model_iv)
            market_ivs.append(market_iv)

        out["model_price"] = model_prices
        out["model_iv"] = model_ivs
        out["market_iv"] = market_ivs
        out["price_error"] = out["model_price"] - out["market_price"]
        out["iv_error"] = out["model_iv"] - out["market_iv"]
        return out
