"""
rBergomi calibration module.

Calibrates model parameters to option cross-section data by minimizing
weighted implied-volatility errors, with optional skew penalty.
"""

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from rBergomiPricer import rBergomiPricer


class rBergomiCalibrator:
    """Calibrator for rBergomi parameters."""

    def __init__(
        self,
        pricer: Optional[rBergomiPricer] = None,
        random_seed: int = 1234
    ):
        self.pricer = pricer if pricer is not None else rBergomiPricer()
        self.random_seed = int(random_seed)

    @staticmethod
    def _validate_columns(option_df: pd.DataFrame) -> None:
        required = {"S0", "r", "q", "maturity", "strike", "option_type", "market_price"}
        missing = required.difference(option_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @staticmethod
    def _compute_log_moneyness(df: pd.DataFrame) -> pd.Series:
        forward = df["S0"] * np.exp((df["r"] - df["q"]) * df["maturity"])
        return np.log(df["strike"] / forward)

    def _price_cross_section_fast(
        self,
        option_df: pd.DataFrame,
        H: float,
        eta: float,
        rho: float,
        xi: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0.04
    ) -> pd.DataFrame:
        """
        Price options by grouping maturities to reduce simulation cost.
        """
        self._validate_columns(option_df)
        out = option_df.copy().reset_index(drop=True)
        out["model_price"] = np.nan
        out["model_iv"] = np.nan
        out["market_iv"] = np.nan

        unique_maturities = np.sort(out["maturity"].unique())

        for i, T in enumerate(unique_maturities):
            sub_idx = out.index[out["maturity"] == T].tolist()
            row0 = out.loc[sub_idx[0]]

            S0 = float(row0["S0"])
            r = float(row0["r"])
            q = float(row0["q"])

            sim = self.pricer.simulate_paths(
                S0=S0,
                T=float(T),
                H=H,
                eta=eta,
                rho=rho,
                xi=xi,
                r=r,
                q=q,
                random_seed=self.random_seed + i
            )
            S_terminal = sim["S"][:, -1]

            for j in sub_idx:
                row = out.loc[j]
                K = float(row["strike"])
                option_type = str(row["option_type"])
                market_price = float(row["market_price"])

                model_price = self.pricer.mc_price_from_terminal(
                    S_terminal=S_terminal,
                    K=K,
                    T=float(T),
                    r=r,
                    option_type=option_type
                )
                model_iv = self.pricer.implied_vol_black(
                    price=model_price, S0=S0, K=K, T=float(T), r=r, q=q, option_type=option_type
                )
                market_iv = self.pricer.implied_vol_black(
                    price=market_price, S0=S0, K=K, T=float(T), r=r, q=q, option_type=option_type
                )

                out.at[j, "model_price"] = model_price
                out.at[j, "model_iv"] = model_iv
                out.at[j, "market_iv"] = market_iv

        out["price_error"] = out["model_price"] - out["market_price"]
        out["iv_error"] = out["model_iv"] - out["market_iv"]
        out["log_moneyness"] = self._compute_log_moneyness(out)
        return out

    @staticmethod
    def _weighted_iv_rmse(priced_df: pd.DataFrame) -> float:
        """
        Weighted IV RMSE by maturity bucket:
        each maturity contributes equally, then averaged across maturities.
        """
        df = priced_df.dropna(subset=["iv_error", "maturity"])
        if df.empty:
            return np.inf

        bucket_rmse = []
        for _, sub in df.groupby("maturity"):
            rmse = np.sqrt(np.mean(np.square(sub["iv_error"].values)))
            if np.isfinite(rmse):
                bucket_rmse.append(rmse)
        if not bucket_rmse:
            return np.inf
        return float(np.mean(bucket_rmse))

    @staticmethod
    def _atm_term_error(priced_df: pd.DataFrame, atm_width: float = 0.05) -> float:
        """Average absolute ATM IV term-structure error."""
        df = priced_df.dropna(subset=["market_iv", "model_iv", "log_moneyness", "maturity"])
        if df.empty:
            return np.nan

        errs = []
        for _, sub in df.groupby("maturity"):
            atm = sub[np.abs(sub["log_moneyness"]) <= atm_width]
            if atm.empty:
                continue
            errs.append(np.abs(atm["iv_error"]).mean())
        if not errs:
            return np.nan
        return float(np.mean(errs))

    @staticmethod
    def _skew_error(priced_df: pd.DataFrame, skew_width: float = 0.15) -> float:
        """
        Average absolute error between model and market ATM skew slopes
        (IV vs log-moneyness regression slope).
        """
        df = priced_df.dropna(subset=["market_iv", "model_iv", "log_moneyness", "maturity"])
        if df.empty:
            return np.nan

        slope_errs = []
        for _, sub in df.groupby("maturity"):
            win = sub[np.abs(sub["log_moneyness"]) <= skew_width]
            if len(win) < 4:
                continue

            x = win["log_moneyness"].values
            mkt = win["market_iv"].values
            mdl = win["model_iv"].values

            mkt_slope, _, _, _, _ = stats.linregress(x, mkt)
            mdl_slope, _, _, _, _ = stats.linregress(x, mdl)
            slope_errs.append(abs(mdl_slope - mkt_slope))

        if not slope_errs:
            return np.nan
        return float(np.mean(slope_errs))

    def compute_fit_metrics(self, priced_df: pd.DataFrame) -> Dict[str, float]:
        """Compute main diagnostics for comparison."""
        iv_rmse = self._weighted_iv_rmse(priced_df)
        atm_term_error = self._atm_term_error(priced_df)
        skew_error = self._skew_error(priced_df)

        valid = priced_df.dropna(subset=["price_error", "iv_error"])
        price_rmse = float(np.sqrt(np.mean(np.square(valid["price_error"].values)))) if not valid.empty else np.nan
        iv_mae = float(np.mean(np.abs(valid["iv_error"].values))) if not valid.empty else np.nan

        return {
            "iv_rmse": iv_rmse,
            "iv_mae": iv_mae,
            "price_rmse": price_rmse,
            "atm_term_error": atm_term_error,
            "skew_error": skew_error
        }

    def calibrate(
        self,
        option_df: pd.DataFrame,
        H_init: float,
        eta_init: float,
        rho_init: float = -0.7,
        xi: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]] = 0.04,
        fix_H: bool = True,
        skew_penalty: float = 0.0,
        bounds_fix_h: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.05, 5.0), (-0.999, -0.01)),
        bounds_free_h: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((0.02, 0.49), (0.05, 5.0), (-0.999, -0.01)),
        maxiter: int = 40
    ) -> Dict[str, object]:
        """
        Calibrate parameters using weighted IV RMSE objective.
        If fix_H=True, optimize (eta, rho).
        Else optimize (H, eta, rho).
        """
        self._validate_columns(option_df)

        if fix_H:
            x0 = np.array([eta_init, rho_init], dtype=float)
            bounds = bounds_fix_h
        else:
            x0 = np.array([H_init, eta_init, rho_init], dtype=float)
            bounds = bounds_free_h

        best_priced_df = None
        best_metrics = None

        def objective(x: np.ndarray) -> float:
            nonlocal best_priced_df, best_metrics

            if fix_H:
                H = float(H_init)
                eta = float(x[0])
                rho = float(x[1])
            else:
                H = float(x[0])
                eta = float(x[1])
                rho = float(x[2])

            priced_df = self._price_cross_section_fast(
                option_df=option_df, H=H, eta=eta, rho=rho, xi=xi
            )
            metrics = self.compute_fit_metrics(priced_df)

            loss = metrics["iv_rmse"]
            if np.isnan(loss) or not np.isfinite(loss):
                return 1e6

            if skew_penalty > 0 and np.isfinite(metrics["skew_error"]):
                loss = loss + skew_penalty * metrics["skew_error"]

            if best_metrics is None or loss < best_metrics["objective"]:
                best_metrics = {**metrics, "objective": float(loss)}
                best_priced_df = priced_df
            return float(loss)

        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter)}
        )

        if fix_H:
            best_H = float(H_init)
            best_eta = float(result.x[0])
            best_rho = float(result.x[1])
        else:
            best_H = float(result.x[0])
            best_eta = float(result.x[1])
            best_rho = float(result.x[2])

        # Ensure we return priced df for optimizer endpoint as well.
        final_priced_df = self._price_cross_section_fast(
            option_df=option_df, H=best_H, eta=best_eta, rho=best_rho, xi=xi
        )
        final_metrics = self.compute_fit_metrics(final_priced_df)

        return {
            "best_params": {
                "H": best_H,
                "eta": best_eta,
                "rho": best_rho
            },
            "initial_params": {
                "H": float(H_init),
                "eta": float(eta_init),
                "rho": float(rho_init)
            },
            "optimizer": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "n_iter": int(result.nit)
            },
            "metrics": final_metrics,
            "priced_df": final_priced_df,
            "best_seen_metrics": best_metrics,
            "best_seen_priced_df": best_priced_df
        }
