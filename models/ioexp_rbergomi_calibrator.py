"""
IO experiment rBergomi calibrator (precision-first objective).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ioexp_rbergomi_pricer import IOExpRBergomiPricer, XiType


@dataclass(frozen=True)
class CalibrationConfig:
    fix_H: bool = True
    skew_penalty: float = 0.15
    atm_term_penalty: float = 0.10
    moneyness_clip: float = 0.30
    maxiter_stage1: int = 30
    maxiter_stage2: int = 50
    bounds_fix_h: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.05, 6.0), (-0.999, -0.01))
    bounds_free_h: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.02, 0.49),
        (0.05, 6.0),
        (-0.999, -0.01),
    )


class IOExpRBergomiCalibrator:
    def __init__(
        self,
        pricer_stage1: Optional[IOExpRBergomiPricer] = None,
        pricer_stage2: Optional[IOExpRBergomiPricer] = None,
        config: CalibrationConfig = CalibrationConfig(),
        random_seed: int = 1234,
    ):
        self.pricer_stage1 = pricer_stage1 if pricer_stage1 is not None else IOExpRBergomiPricer()
        self.pricer_stage2 = pricer_stage2 if pricer_stage2 is not None else IOExpRBergomiPricer()
        self.config = config
        self.random_seed = int(random_seed)

    @staticmethod
    def _validate_columns(option_df: pd.DataFrame) -> None:
        required = {"underlying", "r", "q", "maturity", "strike", "option_type", "market_price"}
        missing = required.difference(option_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @staticmethod
    def _compute_log_moneyness(df: pd.DataFrame) -> pd.Series:
        fwd = df["underlying"] * np.exp((df["r"] - df["q"]) * df["maturity"])
        return np.log(df["strike"] / fwd)

    def _clip_for_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        lm = self._compute_log_moneyness(out)
        out = out[np.abs(lm) <= self.config.moneyness_clip].copy()
        return out.reset_index(drop=True)

    @staticmethod
    def _bucket_weighted_iv_rmse(priced_df: pd.DataFrame) -> float:
        df = priced_df.dropna(subset=["iv_error", "maturity"])
        if df.empty:
            return np.inf
        rmses = []
        for _, sub in df.groupby("maturity"):
            v = np.sqrt(np.mean(np.square(sub["iv_error"].values)))
            if np.isfinite(v):
                rmses.append(v)
        if not rmses:
            return np.inf
        return float(np.mean(rmses))

    @staticmethod
    def _atm_term_error(priced_df: pd.DataFrame, width: float = 0.05) -> float:
        df = priced_df.dropna(subset=["market_iv", "model_iv", "maturity", "log_moneyness"])
        if df.empty:
            return np.nan
        errs = []
        for _, sub in df.groupby("maturity"):
            atm = sub[np.abs(sub["log_moneyness"]) <= width]
            if atm.empty:
                continue
            errs.append(np.mean(np.abs(atm["iv_error"].values)))
        return float(np.mean(errs)) if errs else np.nan

    @staticmethod
    def _skew_error(priced_df: pd.DataFrame, width: float = 0.15) -> float:
        df = priced_df.dropna(subset=["market_iv", "model_iv", "maturity", "log_moneyness"])
        if df.empty:
            return np.nan
        errs = []
        for _, sub in df.groupby("maturity"):
            win = sub[np.abs(sub["log_moneyness"]) <= width]
            if len(win) < 4:
                continue
            x = win["log_moneyness"].to_numpy()
            mkt = win["market_iv"].to_numpy()
            mdl = win["model_iv"].to_numpy()
            mkt_slope, _, _, _, _ = stats.linregress(x, mkt)
            mdl_slope, _, _, _, _ = stats.linregress(x, mdl)
            errs.append(abs(mdl_slope - mkt_slope))
        return float(np.mean(errs)) if errs else np.nan

    def _score(self, priced_df: pd.DataFrame) -> Dict[str, float]:
        iv_rmse = self._bucket_weighted_iv_rmse(priced_df)
        atm_term = self._atm_term_error(priced_df)
        skew = self._skew_error(priced_df)
        loss = iv_rmse
        if np.isfinite(skew):
            loss += self.config.skew_penalty * skew
        if np.isfinite(atm_term):
            loss += self.config.atm_term_penalty * atm_term
        return {
            "objective": float(loss) if np.isfinite(loss) else np.inf,
            "iv_rmse": float(iv_rmse),
            "atm_term_error": float(atm_term) if np.isfinite(atm_term) else np.nan,
            "skew_error": float(skew) if np.isfinite(skew) else np.nan,
        }

    def _objective_factory(
        self,
        option_df: pd.DataFrame,
        H_init: float,
        xi: XiType,
        pricer: IOExpRBergomiPricer,
        maxiter_seed_offset: int,
    ):
        best = {"metrics": None, "priced_df": None}

        def objective(x: np.ndarray) -> float:
            if self.config.fix_H:
                H = float(H_init)
                eta = float(x[0])
                rho = float(x[1])
            else:
                H = float(x[0])
                eta = float(x[1])
                rho = float(x[2])

            priced = pricer.price_cross_section(
                option_df=option_df,
                H=H,
                eta=eta,
                rho=rho,
                xi=xi,
                random_seed=int(self.random_seed + maxiter_seed_offset),
            )
            metrics = self._score(priced)
            if best["metrics"] is None or metrics["objective"] < best["metrics"]["objective"]:
                best["metrics"] = metrics
                best["priced_df"] = priced
            return float(metrics["objective"]) if np.isfinite(metrics["objective"]) else 1e6

        return objective, best

    def calibrate(
        self,
        option_df: pd.DataFrame,
        H_init: float,
        eta_init: float,
        rho_init: float = -0.7,
        xi: XiType = 0.04,
    ) -> Dict[str, object]:
        self._validate_columns(option_df)
        fit_df = self._clip_for_stability(option_df)
        if fit_df.empty:
            raise ValueError("No rows left after moneyness clipping.")

        if self.config.fix_H:
            x0 = np.array([eta_init, rho_init], dtype=float)
            bounds = self.config.bounds_fix_h
        else:
            x0 = np.array([H_init, eta_init, rho_init], dtype=float)
            bounds = self.config.bounds_free_h

        obj1, best1 = self._objective_factory(fit_df, H_init, xi, self.pricer_stage1, maxiter_seed_offset=1)
        res1 = minimize(
            obj1,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(self.config.maxiter_stage1)},
        )

        obj2, best2 = self._objective_factory(fit_df, H_init, xi, self.pricer_stage2, maxiter_seed_offset=101)
        res2 = minimize(
            obj2,
            x0=res1.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(self.config.maxiter_stage2)},
        )

        if self.config.fix_H:
            best_H = float(H_init)
            best_eta = float(res2.x[0])
            best_rho = float(res2.x[1])
        else:
            best_H = float(res2.x[0])
            best_eta = float(res2.x[1])
            best_rho = float(res2.x[2])

        final_priced = self.pricer_stage2.price_cross_section(
            option_df=option_df,
            H=best_H,
            eta=best_eta,
            rho=best_rho,
            xi=xi,
            random_seed=self.random_seed + 999,
        )
        final_metrics = self._score(final_priced)
        final_metrics["price_rmse"] = float(np.sqrt(np.mean(np.square(final_priced["price_error"].dropna().values))))
        final_metrics["iv_mae"] = float(np.mean(np.abs(final_priced["iv_error"].dropna().values)))

        return {
            "best_params": {"H": best_H, "eta": best_eta, "rho": best_rho},
            "initial_params": {"H": float(H_init), "eta": float(eta_init), "rho": float(rho_init)},
            "optimizer_stage1": {
                "success": bool(res1.success),
                "status": int(res1.status),
                "message": str(res1.message),
                "n_iter": int(res1.nit),
            },
            "optimizer_stage2": {
                "success": bool(res2.success),
                "status": int(res2.status),
                "message": str(res2.message),
                "n_iter": int(res2.nit),
            },
            "metrics": final_metrics,
            "priced_df": final_priced,
            "best_seen_stage1": best1["metrics"],
            "best_seen_stage2": best2["metrics"],
        }
