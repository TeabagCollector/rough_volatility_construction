"""
IO experiment data contract and dataset preparation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_CANONICAL_COLUMNS = (
    "trade_date",
    "maturity",
    "strike",
    "option_type",
    "market_price",
    "underlying",
    "r",
    "q",
)


@dataclass(frozen=True)
class IOSliceConfig:
    """Rolling split configuration in trading-day units."""

    train_days: int = 252
    valid_days: int = 63
    test_days: int = 63
    step_days: int = 21
    min_total_days: int = 420


def _normalize_option_type(value: object) -> str:
    s = str(value).strip().lower()
    if s in {"c", "call", "认购", "看涨"}:
        return "call"
    if s in {"p", "put", "认沽", "看跌"}:
        return "put"
    raise ValueError(f"Unsupported option_type value: {value}")


def _ensure_positive(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        out = out[np.isfinite(out[c]) & (out[c] > 0)]
    return out


def standardize_io_columns(
    raw_df: pd.DataFrame,
    column_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Map raw IO columns to canonical schema used across experiment modules.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("raw_df is empty.")

    df = raw_df.copy()
    if column_map:
        df = df.rename(columns=dict(column_map))

    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}")

    out = df.loc[:, list(REQUIRED_CANONICAL_COLUMNS)].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["maturity"] = pd.to_numeric(out["maturity"], errors="coerce")
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["market_price"] = pd.to_numeric(out["market_price"], errors="coerce")
    out["underlying"] = pd.to_numeric(out["underlying"], errors="coerce")
    out["r"] = pd.to_numeric(out["r"], errors="coerce")
    out["q"] = pd.to_numeric(out["q"], errors="coerce")
    out["option_type"] = out["option_type"].map(_normalize_option_type)
    out = out.dropna()
    out = _ensure_positive(out, ["maturity", "strike", "market_price", "underlying"])
    out = out.sort_values(["trade_date", "maturity", "strike", "option_type"]).reset_index(drop=True)
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward and moneyness columns needed by calibration/evaluation.
    """
    out = df.copy()
    out["forward"] = out["underlying"] * np.exp((out["r"] - out["q"]) * out["maturity"])
    out["log_moneyness"] = np.log(out["strike"] / out["forward"])
    out["tau_days"] = np.round(out["maturity"] * 252.0).astype(int)
    return out


def make_day_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add maturity and moneyness buckets for grouped diagnostics.
    """
    out = df.copy()
    tau = out["tau_days"].to_numpy()
    mny = out["log_moneyness"].to_numpy()

    maturity_bucket = np.where(
        tau <= 30,
        "short",
        np.where(tau <= 90, "mid", "long"),
    )
    moneyness_bucket = np.where(
        np.abs(mny) <= 0.03,
        "ATM",
        np.where(mny < -0.03, "ITM_call_OTM_put", "OTM_call_ITM_put"),
    )

    out["maturity_bucket"] = maturity_bucket
    out["moneyness_bucket"] = moneyness_bucket
    return out


def build_canonical_dataset(
    raw_df: pd.DataFrame,
    column_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Full canonical pipeline for IO option cross-section data.
    """
    out = standardize_io_columns(raw_df=raw_df, column_map=column_map)
    out = add_derived_columns(out)
    out = make_day_buckets(out)
    return out


def generate_rolling_slices(
    df: pd.DataFrame,
    config: IOSliceConfig = IOSliceConfig(),
) -> List[Dict[str, pd.Timestamp]]:
    """
    Generate rolling (train/valid/test) date windows.
    """
    if df.empty:
        return []

    unique_days = np.array(sorted(pd.to_datetime(df["trade_date"]).dt.normalize().unique()))
    if unique_days.shape[0] < config.min_total_days:
        return []

    windows: List[Dict[str, pd.Timestamp]] = []
    start = 0
    horizon = config.train_days + config.valid_days + config.test_days

    while start + horizon <= unique_days.shape[0]:
        train_start = unique_days[start]
        train_end = unique_days[start + config.train_days - 1]
        valid_end = unique_days[start + config.train_days + config.valid_days - 1]
        test_end = unique_days[start + horizon - 1]

        windows.append(
            {
                "train_start": pd.Timestamp(train_start),
                "train_end": pd.Timestamp(train_end),
                "valid_start": pd.Timestamp(unique_days[start + config.train_days]),
                "valid_end": pd.Timestamp(valid_end),
                "test_start": pd.Timestamp(unique_days[start + config.train_days + config.valid_days]),
                "test_end": pd.Timestamp(test_end),
            }
        )
        start += config.step_days

    return windows


def select_slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Select rows between [start, end] by trade_date.
    """
    mask = (df["trade_date"] >= pd.Timestamp(start)) & (df["trade_date"] <= pd.Timestamp(end))
    return df.loc[mask].copy().reset_index(drop=True)
