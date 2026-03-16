"""
IO experiment baseline models: BS-flat, local quadratic IV, SABR, Heston.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def _option_sign(option_type: str) -> int:
    s = str(option_type).lower().strip()
    if s in {"call", "c"}:
        return 1
    if s in {"put", "p"}:
        return -1
    raise ValueError(f"Unsupported option type: {option_type}")


def black_price(S0: float, K: float, T: float, r: float, q: float, vol: float, option_type: str) -> float:
    if T <= 0:
        w = _option_sign(option_type)
        return max(w * (S0 - K), 0.0)
    F = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)
    vol = max(float(vol), 1e-10)
    std = vol * np.sqrt(T)
    w = _option_sign(option_type)
    d1 = np.log(F / K) / std + 0.5 * std
    d2 = d1 - std
    return float(disc * (w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)))


def implied_vol_black(price: float, S0: float, K: float, T: float, r: float, q: float, option_type: str) -> float:
    if T <= 0 or price <= 0:
        return np.nan
    low, high = 1e-6, 5.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        p_mid = black_price(S0, K, T, r, q, mid, option_type)
        if p_mid > price:
            high = mid
        else:
            low = mid
    return float(0.5 * (low + high))


def _prepare_market_iv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["market_iv"] = out.apply(
        lambda r: implied_vol_black(
            price=float(r["market_price"]),
            S0=float(r["underlying"]),
            K=float(r["strike"]),
            T=float(r["maturity"]),
            r=float(r["r"]),
            q=float(r["q"]),
            option_type=str(r["option_type"]),
        ),
        axis=1,
    )
    out["forward"] = out["underlying"] * np.exp((out["r"] - out["q"]) * out["maturity"])
    out["log_moneyness"] = np.log(out["strike"] / out["forward"])
    return out


def price_bs_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Per-maturity flat vol fit as lightweight BS baseline."""
    out = _prepare_market_iv(df)
    out["model_iv"] = np.nan
    out["model_price"] = np.nan

    for T, sub in out.groupby("maturity"):
        mkt = sub["market_iv"].dropna().to_numpy()
        if mkt.size == 0:
            continue
        vol_hat = float(np.nanmedian(mkt))
        idx = sub.index
        out.loc[idx, "model_iv"] = vol_hat
        out.loc[idx, "model_price"] = out.loc[idx].apply(
            lambda r: black_price(
                S0=float(r["underlying"]),
                K=float(r["strike"]),
                T=float(r["maturity"]),
                r=float(r["r"]),
                q=float(r["q"]),
                vol=vol_hat,
                option_type=str(r["option_type"]),
            ),
            axis=1,
        )
    out["iv_error"] = out["model_iv"] - out["market_iv"]
    out["price_error"] = out["model_price"] - out["market_price"]
    out["model_name"] = "bs_flat"
    return out


def price_localvol_quadratic(df: pd.DataFrame) -> pd.DataFrame:
    """Quadratic IV smile by maturity as local-vol-like baseline."""
    out = _prepare_market_iv(df)
    out["model_iv"] = np.nan
    out["model_price"] = np.nan

    for T, sub in out.groupby("maturity"):
        clean = sub.dropna(subset=["market_iv", "log_moneyness"])
        if clean.shape[0] < 4:
            continue
        x = clean["log_moneyness"].to_numpy()
        y = clean["market_iv"].to_numpy()
        coeff = np.polyfit(x, y, deg=2)
        idx = sub.index
        x_all = out.loc[idx, "log_moneyness"].to_numpy()
        iv_hat = coeff[0] * x_all ** 2 + coeff[1] * x_all + coeff[2]
        iv_hat = np.clip(iv_hat, 1e-4, 5.0)
        out.loc[idx, "model_iv"] = iv_hat
        out.loc[idx, "model_price"] = out.loc[idx].apply(
            lambda r: black_price(
                S0=float(r["underlying"]),
                K=float(r["strike"]),
                T=float(r["maturity"]),
                r=float(r["r"]),
                q=float(r["q"]),
                vol=float(r["model_iv"]),
                option_type=str(r["option_type"]),
            ),
            axis=1,
        )
    out["iv_error"] = out["model_iv"] - out["market_iv"]
    out["price_error"] = out["model_price"] - out["market_price"]
    out["model_name"] = "localvol_quadratic_iv"
    return out


def sabr_hagan_iv(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    if F <= 0 or K <= 0 or T <= 0:
        return np.nan
    if abs(F - K) < 1e-12:
        fk_beta = F ** (1 - beta)
        term1 = alpha / fk_beta
        term2 = (
            ((1 - beta) ** 2 / 24.0) * (alpha ** 2 / (fk_beta ** 2))
            + 0.25 * rho * beta * nu * alpha / fk_beta
            + (2 - 3 * rho ** 2) * nu ** 2 / 24.0
        ) * T
        return float(max(term1 * (1 + term2), 1e-6))

    log_fk = np.log(F / K)
    fk_beta = (F * K) ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * log_fk
    xz_num = np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho
    xz_den = 1 - rho
    xz = np.log(xz_num / xz_den)
    denom = fk_beta * (1 + ((1 - beta) ** 2 / 24) * log_fk ** 2 + ((1 - beta) ** 4 / 1920) * log_fk ** 4)
    term_time = (
        ((1 - beta) ** 2 / 24.0) * (alpha ** 2 / (fk_beta ** 2))
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2 - 3 * rho ** 2) * nu ** 2 / 24.0
    ) * T
    iv = (alpha / denom) * (z / xz) * (1 + term_time)
    return float(np.clip(iv, 1e-6, 5.0))


@dataclass(frozen=True)
class SABRConfig:
    beta: float = 0.7
    alpha_bounds: Tuple[float, float] = (1e-4, 3.0)
    rho_bounds: Tuple[float, float] = (-0.999, 0.999)
    nu_bounds: Tuple[float, float] = (1e-4, 5.0)
    maxiter: int = 100


def price_sabr(df: pd.DataFrame, cfg: SABRConfig = SABRConfig()) -> pd.DataFrame:
    out = _prepare_market_iv(df)
    out["model_iv"] = np.nan
    out["model_price"] = np.nan

    for T, sub in out.groupby("maturity"):
        clean = sub.dropna(subset=["market_iv"])
        if clean.shape[0] < 5:
            continue
        F = float(np.nanmedian(clean["forward"]))
        ks = clean["strike"].to_numpy()
        iv_mkt = clean["market_iv"].to_numpy()

        def obj(x: np.ndarray) -> float:
            alpha, rho, nu = float(x[0]), float(x[1]), float(x[2])
            iv_hat = np.array([sabr_hagan_iv(F, k, float(T), alpha, cfg.beta, rho, nu) for k in ks])
            if np.any(~np.isfinite(iv_hat)):
                return 1e6
            return float(np.mean((iv_hat - iv_mkt) ** 2))

        x0 = np.array([0.2, -0.2, 0.7], dtype=float)
        res = minimize(
            obj,
            x0=x0,
            method="L-BFGS-B",
            bounds=[cfg.alpha_bounds, cfg.rho_bounds, cfg.nu_bounds],
            options={"maxiter": int(cfg.maxiter)},
        )
        alpha_hat, rho_hat, nu_hat = res.x
        idx = sub.index
        iv_hat_all = np.array(
            [
                sabr_hagan_iv(
                    float(out.at[i, "forward"]),
                    float(out.at[i, "strike"]),
                    float(out.at[i, "maturity"]),
                    float(alpha_hat),
                    cfg.beta,
                    float(rho_hat),
                    float(nu_hat),
                )
                for i in idx
            ]
        )
        out.loc[idx, "model_iv"] = iv_hat_all
        out.loc[idx, "model_price"] = out.loc[idx].apply(
            lambda r: black_price(
                S0=float(r["underlying"]),
                K=float(r["strike"]),
                T=float(r["maturity"]),
                r=float(r["r"]),
                q=float(r["q"]),
                vol=float(r["model_iv"]),
                option_type=str(r["option_type"]),
            ),
            axis=1,
        )

    out["iv_error"] = out["model_iv"] - out["market_iv"]
    out["price_error"] = out["model_price"] - out["market_price"]
    out["model_name"] = "sabr_hagan"
    return out


def heston_charfunc(u: complex, T: float, S0: float, r: float, q: float, params: np.ndarray) -> complex:
    kappa, theta, sigma, rho, v0 = params
    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma ** 2 * (iu + u ** 2))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    exp_dt = np.exp(-d * T)
    c = (
        iu * (np.log(S0) + (r - q) * T)
        + (kappa * theta / sigma ** 2) * ((kappa - rho * sigma * iu - d) * T - 2.0 * np.log((1 - g * exp_dt) / (1 - g)))
    )
    d_term = ((kappa - rho * sigma * iu - d) / sigma ** 2) * ((1 - exp_dt) / (1 - g * exp_dt))
    return np.exp(c + d_term * v0)


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: np.ndarray,
    u_max: float = 100.0,
    n_u: int = 400,
) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)

    ln_k = np.log(K)
    phi_minus_i = heston_charfunc(-1j, T, S0, r, q, params)
    us = np.linspace(1e-6, u_max, n_u)

    def p1_integrand(u: float) -> float:
        cf = heston_charfunc(u - 1j, T, S0, r, q, params)
        numer = np.exp(-1j * u * ln_k) * cf
        denom = 1j * u * phi_minus_i
        return np.real(numer / denom)

    def p2_integrand(u: float) -> float:
        cf = heston_charfunc(u, T, S0, r, q, params)
        numer = np.exp(-1j * u * ln_k) * cf
        denom = 1j * u
        return np.real(numer / denom)

    p1 = 0.5 + (1.0 / np.pi) * np.trapz(np.vectorize(p1_integrand)(us), us)
    p2 = 0.5 + (1.0 / np.pi) * np.trapz(np.vectorize(p2_integrand)(us), us)
    return float(S0 * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2)


def price_heston(df: pd.DataFrame, maxiter: int = 40) -> pd.DataFrame:
    out = _prepare_market_iv(df)
    out["model_iv"] = np.nan
    out["model_price"] = np.nan

    def loss_fn(x: np.ndarray) -> float:
        kappa, theta, sigma, rho, v0 = x
        if sigma <= 0 or theta <= 0 or v0 <= 0 or kappa <= 0 or abs(rho) >= 1:
            return 1e6
        params = np.array([kappa, theta, sigma, rho, v0], dtype=float)
        model_prices = []
        for _, row in out.iterrows():
            p = heston_call_price(
                S0=float(row["underlying"]),
                K=float(row["strike"]),
                T=float(row["maturity"]),
                r=float(row["r"]),
                q=float(row["q"]),
                params=params,
            )
            if row["option_type"] == "put":
                p = p - float(row["underlying"]) * np.exp(-float(row["q"]) * float(row["maturity"])) + float(
                    row["strike"]
                ) * np.exp(-float(row["r"]) * float(row["maturity"]))
            model_prices.append(p)
        model_prices = np.array(model_prices)
        mask = np.isfinite(model_prices) & np.isfinite(out["market_price"].to_numpy())
        if mask.sum() < 5:
            return 1e6
        return float(np.mean((model_prices[mask] - out["market_price"].to_numpy()[mask]) ** 2))

    x0 = np.array([1.2, 0.04, 0.5, -0.6, 0.04], dtype=float)
    bounds = [(0.05, 8.0), (1e-4, 2.0), (0.05, 3.0), (-0.999, 0.999), (1e-4, 2.0)]
    res = minimize(loss_fn, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
    params = np.array(res.x, dtype=float)

    prices = []
    ivs = []
    for _, row in out.iterrows():
        p = heston_call_price(
            S0=float(row["underlying"]),
            K=float(row["strike"]),
            T=float(row["maturity"]),
            r=float(row["r"]),
            q=float(row["q"]),
            params=params,
        )
        if row["option_type"] == "put":
            p = p - float(row["underlying"]) * np.exp(-float(row["q"]) * float(row["maturity"])) + float(
                row["strike"]
            ) * np.exp(-float(row["r"]) * float(row["maturity"]))
        prices.append(p)
        ivs.append(
            implied_vol_black(
                price=float(p),
                S0=float(row["underlying"]),
                K=float(row["strike"]),
                T=float(row["maturity"]),
                r=float(row["r"]),
                q=float(row["q"]),
                option_type=str(row["option_type"]),
            )
        )
    out["model_price"] = np.array(prices)
    out["model_iv"] = np.array(ivs)
    out["iv_error"] = out["model_iv"] - out["market_iv"]
    out["price_error"] = out["model_price"] - out["market_price"]
    out["model_name"] = "heston_cf"
    return out


def run_all_baselines(option_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "bs_flat": price_bs_flat(option_df),
        "localvol_quadratic_iv": price_localvol_quadratic(option_df),
        "sabr_hagan": price_sabr(option_df),
        "heston_cf": price_heston(option_df),
    }
