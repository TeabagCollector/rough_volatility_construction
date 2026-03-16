"""
Unified evaluation metrics for IO experiment models.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_rmse(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(x))))


def _safe_mae(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x)))


def compute_core_metrics(priced_df: pd.DataFrame) -> Dict[str, float]:
    df = priced_df.dropna(subset=["iv_error", "price_error"]).copy()
    if df.empty:
        return {
            "iv_rmse": np.nan,
            "iv_mae": np.nan,
            "price_rmse": np.nan,
            "price_mae": np.nan,
            "n": 0,
        }
    iv_err = df["iv_error"].to_numpy()
    px_err = df["price_error"].to_numpy()
    return {
        "iv_rmse": _safe_rmse(iv_err),
        "iv_mae": _safe_mae(iv_err),
        "price_rmse": _safe_rmse(px_err),
        "price_mae": _safe_mae(px_err),
        "n": int(df.shape[0]),
    }


def compute_group_metrics(priced_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for g, sub in priced_df.groupby(group_col):
        met = compute_core_metrics(sub)
        rows.append({"group": g, **met})
    if not rows:
        return pd.DataFrame(columns=["group", "iv_rmse", "iv_mae", "price_rmse", "price_mae", "n"])
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compute_skew_error(priced_df: pd.DataFrame, width: float = 0.15) -> float:
    df = priced_df.dropna(subset=["market_iv", "model_iv", "log_moneyness", "maturity"]).copy()
    if df.empty:
        return np.nan
    errs = []
    for _, sub in df.groupby("maturity"):
        win = sub[np.abs(sub["log_moneyness"]) <= width]
        if win.shape[0] < 4:
            continue
        x = win["log_moneyness"].to_numpy()
        mkt = win["market_iv"].to_numpy()
        mdl = win["model_iv"].to_numpy()
        mkt_slope = np.polyfit(x, mkt, deg=1)[0]
        mdl_slope = np.polyfit(x, mdl, deg=1)[0]
        errs.append(abs(mdl_slope - mkt_slope))
    return float(np.mean(errs)) if errs else np.nan


def compute_atm_term_error(priced_df: pd.DataFrame, atm_width: float = 0.05) -> float:
    df = priced_df.dropna(subset=["market_iv", "model_iv", "maturity", "log_moneyness"]).copy()
    if df.empty:
        return np.nan
    errs = []
    for _, sub in df.groupby("maturity"):
        atm = sub[np.abs(sub["log_moneyness"]) <= atm_width]
        if atm.empty:
            continue
        errs.append(np.mean(np.abs((atm["model_iv"] - atm["market_iv"]).to_numpy())))
    return float(np.mean(errs)) if errs else np.nan


def summarize_model(priced_df: pd.DataFrame, model_name: str) -> Dict[str, float]:
    core = compute_core_metrics(priced_df)
    core["skew_error"] = compute_skew_error(priced_df)
    core["atm_term_error"] = compute_atm_term_error(priced_df)
    core["model"] = model_name
    return core
