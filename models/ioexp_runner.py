"""
End-to-end runner for IO experiment models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ioexp_baselines import run_all_baselines
from ioexp_data_contract import (
    IOSliceConfig,
    build_canonical_dataset,
    generate_rolling_slices,
    select_slice,
)
from ioexp_data_filters import IOFilterConfig, apply_all_filters
from ioexp_eval_metrics import summarize_model
from ioexp_rbergomi_calibrator import CalibrationConfig, IOExpRBergomiCalibrator
from ioexp_rbergomi_pricer import IOExpRBergomiPricer, MCConfig
from ioexp_rfsv_bridge import (
    RFSVPriorConfig,
    attach_atm_rfsv_reference,
    build_xi_prior_from_rfsv,
    load_rfsv_predictions,
)


@dataclass(frozen=True)
class RunnerConfig:
    data: IOSliceConfig = IOSliceConfig()
    filters: IOFilterConfig = IOFilterConfig()
    mc_stage1: MCConfig = MCConfig(n_paths=8000, antithetic=True, use_control_variate=True)
    mc_stage2: MCConfig = MCConfig(n_paths=24000, antithetic=True, use_control_variate=True)
    calib: CalibrationConfig = CalibrationConfig()
    rfsv_prior: RFSVPriorConfig = RFSVPriorConfig()


class IOExperimentRunner:
    def __init__(self, cfg: RunnerConfig = RunnerConfig()):
        self.cfg = cfg
        self.pricer1 = IOExpRBergomiPricer(config=cfg.mc_stage1)
        self.pricer2 = IOExpRBergomiPricer(config=cfg.mc_stage2)
        self.calibrator = IOExpRBergomiCalibrator(
            pricer_stage1=self.pricer1,
            pricer_stage2=self.pricer2,
            config=cfg.calib,
        )

    def prepare_dataset(self, raw_df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        canon = build_canonical_dataset(raw_df=raw_df, column_map=column_map)
        filtered, _ = apply_all_filters(canon, cfg=self.cfg.filters)
        return filtered

    def run_single_test_window(
        self,
        prepared_df: pd.DataFrame,
        rfsv_pred_path: Optional[str] = None,
        H_init: float = 0.1,
        eta_init: float = 1.0,
        rho_init: float = -0.7,
    ) -> Dict[str, object]:
        windows = generate_rolling_slices(prepared_df, config=self.cfg.data)
        if not windows:
            raise ValueError("No rolling windows generated. Check dataset length.")
        win = windows[-1]
        train_df = select_slice(prepared_df, win["train_start"], win["train_end"])
        test_df = select_slice(prepared_df, win["test_start"], win["test_end"])
        if train_df.empty or test_df.empty:
            raise ValueError("Train/test split is empty.")

        xi_prior = 0.04
        rfsv_df = None
        if rfsv_pred_path:
            rfsv_df = load_rfsv_predictions(rfsv_pred_path)
            xi_prior = build_xi_prior_from_rfsv(
                rfsv_df=rfsv_df,
                trade_date=pd.Timestamp(test_df["trade_date"].iloc[0]),
                T=float(np.nanmedian(test_df["maturity"])),
                cfg=self.cfg.rfsv_prior,
            )

        # rBergomi plain (no RFSV prior)
        calib_plain = self.calibrator.calibrate(
            option_df=train_df,
            H_init=H_init,
            eta_init=eta_init,
            rho_init=rho_init,
            xi=0.04,
        )
        best_plain = calib_plain["best_params"]
        rb_test_plain = self.pricer2.price_cross_section(
            option_df=test_df,
            H=float(best_plain["H"]),
            eta=float(best_plain["eta"]),
            rho=float(best_plain["rho"]),
            xi=0.04,
            random_seed=2026,
        )
        rb_test_plain["model_name"] = "rbergomi_plain"

        # rBergomi with RFSV prior
        calib_prior = self.calibrator.calibrate(
            option_df=train_df,
            H_init=H_init,
            eta_init=eta_init,
            rho_init=rho_init,
            xi=xi_prior,
        )
        best_prior = calib_prior["best_params"]
        rb_test_prior = self.pricer2.price_cross_section(
            option_df=test_df,
            H=float(best_prior["H"]),
            eta=float(best_prior["eta"]),
            rho=float(best_prior["rho"]),
            xi=xi_prior,
            random_seed=2027,
        )
        rb_test_prior["model_name"] = "rbergomi_with_rfsv_prior"

        baseline_map = run_all_baselines(test_df)
        summary_rows = [
            summarize_model(rb_test_plain, "rbergomi_plain"),
            summarize_model(rb_test_prior, "rbergomi_with_rfsv_prior"),
        ]
        for name, df_b in baseline_map.items():
            summary_rows.append(summarize_model(df_b, name))
        summary = pd.DataFrame(summary_rows).sort_values("iv_rmse").reset_index(drop=True)

        atm_comparison = None
        if rfsv_df is not None:
            atm_ref = attach_atm_rfsv_reference(test_df, rfsv_df, trade_date_col="trade_date")
            atm_mask = np.abs(atm_ref["log_moneyness"]) <= 0.03
            atm_ref = atm_ref.loc[atm_mask].copy()
            if not atm_ref.empty:
                atm_ref["rfsv_ref_iv_proxy"] = np.sqrt(np.maximum(atm_ref["rfsv_ref_var"], 1e-12))
                plain_atm = rb_test_plain.loc[atm_ref.index, "model_iv"].to_numpy()
                prior_atm = rb_test_prior.loc[atm_ref.index, "model_iv"].to_numpy()
                rfsv_atm = atm_ref["rfsv_ref_iv_proxy"].to_numpy()
                market_atm = rb_test_prior.loc[atm_ref.index, "market_iv"].to_numpy()
                atm_comparison = pd.DataFrame(
                    {
                        "market_iv": market_atm,
                        "rfsv_atm_proxy": rfsv_atm,
                        "rbergomi_plain_iv": plain_atm,
                        "rbergomi_prior_iv": prior_atm,
                    }
                )

        return {
            "window": win,
            "train_rows": int(train_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "calibration_plain": calib_plain,
            "calibration_with_rfsv_prior": calib_prior,
            "rbergomi_test_df_plain": rb_test_plain,
            "rbergomi_test_df_with_rfsv_prior": rb_test_prior,
            "baseline_results": baseline_map,
            "atm_comparison": atm_comparison,
            "summary": summary,
        }
