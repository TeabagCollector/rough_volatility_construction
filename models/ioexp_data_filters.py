"""
IO experiment data quality filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IOFilterConfig:
    min_price: float = 0.2
    max_rel_spread: float = 0.35
    min_maturity_days: int = 5
    max_maturity_days: int = 365
    max_abs_log_moneyness: float = 0.35


def _option_sign(option_type: str) -> int:
    s = str(option_type).strip().lower()
    if s in {"call", "c"}:
        return 1
    if s in {"put", "p"}:
        return -1
    raise ValueError(f"unsupported option_type: {option_type}")


def _intrinsic_bound(row: pd.Series) -> float:
    w = _option_sign(row["option_type"])
    return max(w * (float(row["underlying"]) - float(row["strike"])), 0.0)


def filter_by_static_rules(
    df: pd.DataFrame,
    cfg: IOFilterConfig = IOFilterConfig(),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply static filters independent from model outputs.
    """
    if df.empty:
        return df.copy(), {"input_rows": 0, "output_rows": 0}

    out = df.copy()
    stats = {"input_rows": int(out.shape[0])}

    out = out[np.isfinite(out["market_price"]) & (out["market_price"] >= cfg.min_price)]
    out = out[np.isfinite(out["tau_days"])]
    out = out[(out["tau_days"] >= cfg.min_maturity_days) & (out["tau_days"] <= cfg.max_maturity_days)]
    out = out[np.abs(out["log_moneyness"]) <= cfg.max_abs_log_moneyness]
    out = out.reset_index(drop=True)

    stats["after_static"] = int(out.shape[0])
    return out, stats


def filter_by_microstructure(
    df: pd.DataFrame,
    bid_col: str = "bid",
    ask_col: str = "ask",
    cfg: IOFilterConfig = IOFilterConfig(),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Optional quote-level filter when bid/ask is available.
    """
    if df.empty:
        return df.copy(), {"input_rows": 0, "output_rows": 0}
    if bid_col not in df.columns or ask_col not in df.columns:
        return df.copy(), {"input_rows": int(df.shape[0]), "output_rows": int(df.shape[0])}

    out = df.copy()
    spread = out[ask_col] - out[bid_col]
    mid = 0.5 * (out[ask_col] + out[bid_col])
    rel_spread = spread / np.maximum(mid, 1e-8)
    mask = np.isfinite(rel_spread) & (rel_spread >= 0.0) & (rel_spread <= cfg.max_rel_spread)
    out = out.loc[mask].copy().reset_index(drop=True)
    return out, {"input_rows": int(df.shape[0]), "output_rows": int(out.shape[0])}


def filter_by_no_arbitrage(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Basic no-arbitrage lower-bound checks on option prices.
    """
    if df.empty:
        return df.copy(), {"input_rows": 0, "output_rows": 0}

    out = df.copy()
    intrinsic = out.apply(_intrinsic_bound, axis=1)
    out = out[out["market_price"] + 1e-10 >= intrinsic]
    out = out.reset_index(drop=True)
    return out, {"input_rows": int(df.shape[0]), "output_rows": int(out.shape[0])}


def apply_all_filters(
    df: pd.DataFrame,
    cfg: IOFilterConfig = IOFilterConfig(),
    bid_col: str = "bid",
    ask_col: str = "ask",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Combined IO filtering pipeline.
    """
    out, s_static = filter_by_static_rules(df=df, cfg=cfg)
    out, s_micro = filter_by_microstructure(df=out, bid_col=bid_col, ask_col=ask_col, cfg=cfg)
    out, s_arb = filter_by_no_arbitrage(df=out)
    stats = {
        "input_rows": int(df.shape[0]),
        "after_static": int(s_static.get("after_static", out.shape[0])),
        "after_micro": int(s_micro.get("output_rows", out.shape[0])),
        "after_no_arb": int(s_arb.get("output_rows", out.shape[0])),
    }
    return out, stats
