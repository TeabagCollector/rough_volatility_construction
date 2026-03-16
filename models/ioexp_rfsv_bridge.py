"""
Bridge utilities: connect existing RFSV outputs to IO experiment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RFSVPriorConfig:
    default_xi: float = 0.04
    min_xi: float = 1e-6
    max_xi: float = 2.0


def load_rfsv_predictions(path: str) -> pd.DataFrame:
    """
    Expected columns:
    trade_date, horizon_days, pred_log_var (or pred_var).
    """
    df = pd.read_csv(path)
    if "trade_date" not in df.columns:
        raise ValueError("RFSV prediction file must include trade_date.")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "horizon_days" not in df.columns:
        df["horizon_days"] = 1
    if "pred_var" not in df.columns:
        if "pred_log_var" not in df.columns:
            raise ValueError("Need pred_var or pred_log_var in RFSV prediction file.")
        df["pred_var"] = np.exp(pd.to_numeric(df["pred_log_var"], errors="coerce"))
    df["pred_var"] = pd.to_numeric(df["pred_var"], errors="coerce")
    return df.dropna(subset=["trade_date", "horizon_days", "pred_var"]).reset_index(drop=True)


def build_xi_prior_from_rfsv(
    rfsv_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    T: float,
    cfg: RFSVPriorConfig = RFSVPriorConfig(),
) -> float:
    """
    Scalar xi prior for a target maturity by nearest horizon mapping.
    """
    if rfsv_df.empty:
        return float(cfg.default_xi)
    day = pd.Timestamp(trade_date).normalize()
    sub = rfsv_df[rfsv_df["trade_date"].dt.normalize() == day]
    if sub.empty:
        return float(cfg.default_xi)

    target_h = max(1, int(round(float(T) * 252.0)))
    idx = (sub["horizon_days"] - target_h).abs().idxmin()
    xi = float(sub.at[idx, "pred_var"])
    return float(np.clip(xi, cfg.min_xi, cfg.max_xi))


def make_xi_prior_function(
    rfsv_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    cfg: RFSVPriorConfig = RFSVPriorConfig(),
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build xi(t) callable consumed by rBergomi pricer.
    """
    sub = rfsv_df[rfsv_df["trade_date"].dt.normalize() == pd.Timestamp(trade_date).normalize()].copy()
    if sub.empty:
        return lambda t: np.full_like(t, float(cfg.default_xi), dtype=float)

    sub["T"] = sub["horizon_days"] / 252.0
    sub = sub.sort_values("T")
    t_nodes = sub["T"].to_numpy()
    xi_nodes = np.clip(sub["pred_var"].to_numpy(), cfg.min_xi, cfg.max_xi)

    def xi_func(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        vals = np.interp(t, t_nodes, xi_nodes, left=xi_nodes[0], right=xi_nodes[-1])
        return np.asarray(vals, dtype=float)

    return xi_func


def attach_atm_rfsv_reference(
    option_df: pd.DataFrame,
    rfsv_df: pd.DataFrame,
    trade_date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    Attach closest-horizon RFSV variance as ATM benchmark reference.
    """
    out = option_df.copy()
    out["rfsv_ref_var"] = np.nan
    out["rfsv_ref_log_var"] = np.nan
    for idx, row in out.iterrows():
        day = pd.Timestamp(row[trade_date_col]).normalize()
        T = float(row["maturity"])
        xi = build_xi_prior_from_rfsv(rfsv_df=rfsv_df, trade_date=day, T=T)
        out.at[idx, "rfsv_ref_var"] = xi
        out.at[idx, "rfsv_ref_log_var"] = np.log(max(xi, 1e-12))
    return out
