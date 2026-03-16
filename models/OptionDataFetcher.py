"""
Option data fetcher (split from DataFetcher).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import warnings

import rqdatac as rq

from DataFetcher import UnderlyingDataFetcher

warnings.filterwarnings("ignore")


class OptionDataFetcher(UnderlyingDataFetcher):
    """专门负责期权链路：拉取、清洗、落盘。"""

    _CANONICAL_COLUMNS = [
        "trade_date",
        "maturity",
        "strike",
        "option_type",
        "market_price",
        "underlying",
        "r",
        "q",
    ]
    _EXTRA_COLUMNS = ["order_book_id", "volume", "days_to_expiry", "S0", "api_iv", "log_moneyness"]
    _INSTRUMENT_REQUIRED = [
        "order_book_id",
        "underlying_order_book_id",
        "option_type",
        "strike_price",
        "listed_date",
        "maturity_date",
        "de_listed_date",
    ]

    @staticmethod
    def _pick_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _normalize_option_type(value: Union[str, int, float]) -> Optional[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (int, np.integer)):
            if int(value) == 1:
                return "call"
            if int(value) == -1:
                return "put"
        text = str(value).strip().lower()
        if text in {"c", "call", "认购", "购"}:
            return "call"
        if text in {"p", "put", "认沽", "沽"}:
            return "put"
        return None

    def _align_option_panel_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对齐 ioexp canonical 列，并保留常用附加列。
        """
        if df is None:
            return pd.DataFrame(columns=self._CANONICAL_COLUMNS + self._EXTRA_COLUMNS)
        out = df.copy()

        # 字段兼容：underlying/S0 二选一时自动补齐
        if "underlying" not in out.columns and "S0" in out.columns:
            out["underlying"] = out["S0"]
        if "S0" not in out.columns and "underlying" in out.columns:
            out["S0"] = out["underlying"]

        for col in self._CANONICAL_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        for col in self._EXTRA_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        if "trade_date" in out.columns:
            out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
        for c in ["maturity", "strike", "market_price", "underlying", "r", "q", "volume", "days_to_expiry", "S0"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        if "option_type" in out.columns:
            out["option_type"] = out["option_type"].map(self._normalize_option_type)

        keep = self._CANONICAL_COLUMNS + [c for c in self._EXTRA_COLUMNS if c in out.columns]
        return out.loc[:, keep]

    def get_option_instruments(self, underlying_symbol: str, date: str) -> pd.DataFrame:
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        try:
            df_all = rq.all_instruments(type="Option", date=date)
        except Exception as e:
            print(f"✗ 获取期权合约清单失败: {e}")
            return pd.DataFrame()

        if df_all is None or len(df_all) == 0:
            return pd.DataFrame()

        df_all = df_all.copy()
        missing_cols = [c for c in self._INSTRUMENT_REQUIRED if c not in df_all.columns]
        if missing_cols:
            return pd.DataFrame()

        mask = df_all["underlying_order_book_id"].astype(str) == str(underlying_symbol)
        df_opt = df_all.loc[mask].copy()
        if df_opt.empty:
            return df_opt

        asof_date = pd.to_datetime(date, errors="coerce")
        if pd.isna(asof_date):
            return pd.DataFrame()

        listed_dt = pd.to_datetime(df_opt["listed_date"], errors="coerce")
        delisted_dt = pd.to_datetime(df_opt["de_listed_date"], errors="coerce")
        df_opt = df_opt[(listed_dt.isna()) | (listed_dt <= asof_date)]
        df_opt = df_opt[(delisted_dt.isna()) | (delisted_dt >= asof_date)]
        return df_opt.reset_index(drop=True)

    def get_option_prices(
        self,
        order_book_ids: List[str],
        date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        if fields is None:
            fields = ["open", "high", "low", "close", "volume", "open_interest"]
        if not order_book_ids:
            return pd.DataFrame()

        px = rq.get_price(
            order_book_ids=list(order_book_ids),
            start_date=date,
            end_date=date,
            frequency="1d",
            fields=fields,
            adjust_type="none",
            expect_df=True,
        )
        if px is None or len(px) == 0:
            return pd.DataFrame()

        if isinstance(px, pd.Series):
            px = px.to_frame(name=fields[0] if fields else "value")
        if not isinstance(px, pd.DataFrame):
            return pd.DataFrame()

        df_px = px.copy()
        if isinstance(df_px.index, pd.MultiIndex):
            df_px = df_px.reset_index()
            oid_col = self._pick_existing_column(df_px, ["order_book_id"])
            if oid_col is None:
                return pd.DataFrame()
            df_px[oid_col] = df_px[oid_col].astype(str)
            # 同一日有重复记录时取最后一条
            df_px = df_px.sort_values(by=[oid_col]).drop_duplicates(subset=[oid_col], keep="last")
            if oid_col != "order_book_id":
                df_px = df_px.rename(columns={oid_col: "order_book_id"})
            return df_px.reset_index(drop=True)

        # 非 MultiIndex 情况（通常是单个合约）
        if len(order_book_ids) == 1:
            row = df_px.iloc[-1:].copy()
            row["order_book_id"] = str(order_book_ids[0])
            return row.reset_index(drop=True)
        return pd.DataFrame()

    def get_option_api_iv(
        self,
        order_book_ids: List[str],
        date: str,
        model: str = "implied_forward",
        price_type: str = "close",
        frequency: str = "1d",
    ) -> pd.DataFrame:
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        if not order_book_ids:
            return pd.DataFrame(columns=["order_book_id", "api_iv"])
        try:
            greeks = rq.options.get_greeks(
                order_book_ids=order_book_ids,
                start_date=date,
                end_date=date,
                fields=["iv"],
                model=model,
                price_type=price_type,
                frequency=frequency,
                market="cn",
            )
        except Exception as e:
            print(f"⚠ 获取米筐 iv 失败: {e}")
            return pd.DataFrame(columns=["order_book_id", "api_iv"])

        if greeks is None or len(greeks) == 0:
            return pd.DataFrame(columns=["order_book_id", "api_iv"])
        if isinstance(greeks, pd.Series):
            greeks = greeks.to_frame(name="iv")
        if not isinstance(greeks, pd.DataFrame):
            return pd.DataFrame(columns=["order_book_id", "api_iv"])

        df = greeks.copy()
        if isinstance(df.index, pd.MultiIndex):
            idx_names = list(df.index.names)
            if len(idx_names) >= 2:
                df = df.reset_index()
                oid_col = "order_book_id" if "order_book_id" in df.columns else idx_names[0]
                iv_col = "iv" if "iv" in df.columns else self._pick_existing_column(df, ["iv", "IV"])
                if iv_col is None:
                    return pd.DataFrame(columns=["order_book_id", "api_iv"])
                out = df[[oid_col, iv_col]].rename(columns={oid_col: "order_book_id", iv_col: "api_iv"})
                out = out.dropna(subset=["order_book_id"])
                out["order_book_id"] = out["order_book_id"].astype(str)
                return out.drop_duplicates(subset=["order_book_id"], keep="last").reset_index(drop=True)

        df = df.reset_index()
        oid_col = self._pick_existing_column(df, ["order_book_id", "index"])
        iv_col = self._pick_existing_column(df, ["iv", "IV"])
        if oid_col is None or iv_col is None:
            return pd.DataFrame(columns=["order_book_id", "api_iv"])
        out = df[[oid_col, iv_col]].rename(columns={oid_col: "order_book_id", iv_col: "api_iv"})
        out = out.dropna(subset=["order_book_id"])
        out["order_book_id"] = out["order_book_id"].astype(str)
        return out.drop_duplicates(subset=["order_book_id"], keep="last").reset_index(drop=True)

    def inspect_option_schema(
        self,
        underlying_order_book_id: str,
        date: str,
        sample_size: int = 10,
        price_fields: Optional[List[str]] = None,
    ) -> dict:
        """
        轻量检查当前账号/API返回的真实列名，便于后续固定字段映射。
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")

        if price_fields is None:
            price_fields = ["open", "high", "low", "close", "volume", "open_interest"]

        inst = self.get_option_instruments(underlying_symbol=underlying_order_book_id, date=date)
        option_ids = inst["order_book_id"].astype(str).tolist() if ("order_book_id" in inst.columns and not inst.empty) else []
        sample_ids = option_ids[: max(1, int(sample_size))]
        px = self.get_option_prices(order_book_ids=sample_ids, date=date, fields=price_fields)
        spot = self.get_underlying_daily_price(
            symbol=underlying_order_book_id,
            start_date=date,
            end_date=date,
            fields=["open", "close"],
        )

        info = {
            "date": str(date),
            "instrument_columns": inst.columns.tolist() if isinstance(inst, pd.DataFrame) else [],
            "price_columns": px.columns.tolist() if isinstance(px, pd.DataFrame) else [],
            "underlying_price_columns": spot.columns.tolist() if isinstance(spot, pd.DataFrame) else [],
            "n_contracts": int(len(option_ids)),
            "n_sampled_contracts": int(len(sample_ids)),
        }
        print("[inspect_option_schema]", info)
        return info

    def build_option_cross_section(
        self,
        underlying_symbol: str,
        trade_date: str,
        risk_free_rate: float = 0.02,
        dividend_yield: float = 0.0,
        min_days_to_expiry: int = 5,
        min_price: float = 1e-4,
        min_volume: float = 0.0,
        use_mid_price: bool = True,
        include_api_iv: bool = False,
        greeks_model: str = "implied_forward",
        greeks_price_type: str = "close",
        return_reason: bool = False,
        spot_override: Optional[float] = None,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, str]]:
        spot = float(spot_override) if spot_override is not None else self.get_underlying_spot_on_date(underlying_symbol, trade_date)
        if spot is None or spot <= 0:
            empty = pd.DataFrame()
            return (empty, "underlying_spot_missing_or_nonpositive") if return_reason else empty

        df_inst = self.get_option_instruments(underlying_symbol, trade_date)
        if df_inst.empty:
            empty = pd.DataFrame()
            return (empty, "option_instruments_empty") if return_reason else empty

        required = ["order_book_id", "strike_price", "maturity_date", "option_type"]
        if any(c not in df_inst.columns for c in required):
            empty = pd.DataFrame()
            return (empty, "instrument_required_columns_missing") if return_reason else empty

        option_ids = df_inst["order_book_id"].astype(str).tolist()
        df_px = self.get_option_prices(option_ids, trade_date)
        if df_px.empty:
            empty = pd.DataFrame()
            return (empty, "option_prices_empty") if return_reason else empty

        px_map = df_px.set_index("order_book_id")
        trade_dt = pd.to_datetime(trade_date)
        output = []
        for _, row in df_inst.iterrows():
            oid = str(row["order_book_id"])
            if oid not in px_map.index:
                continue
            strike = float(row["strike_price"]) if pd.notna(row["strike_price"]) else np.nan
            maturity_dt = pd.to_datetime(row["maturity_date"]) if pd.notna(row["maturity_date"]) else pd.NaT
            cp = self._normalize_option_type(row["option_type"])
            if np.isnan(strike) or pd.isna(maturity_dt) or cp is None:
                continue
            days = int((maturity_dt - trade_dt).days)
            if days < min_days_to_expiry:
                continue
            maturity = days / 365.0

            px_row = px_map.loc[oid]
            if isinstance(px_row, pd.DataFrame):
                px_row = px_row.iloc[-1]
            close_px = float(px_row.get("close", np.nan)) if pd.notna(px_row.get("close", np.nan)) else np.nan
            open_px = float(px_row.get("open", np.nan)) if pd.notna(px_row.get("open", np.nan)) else np.nan
            volume = float(px_row.get("volume", 0.0)) if pd.notna(px_row.get("volume", 0.0)) else 0.0
            market_price = 0.5 * (open_px + close_px) if (use_mid_price and not np.isnan(open_px) and not np.isnan(close_px)) else close_px
            if np.isnan(market_price) or market_price < min_price or volume < min_volume:
                continue

            output.append(
                {
                    "trade_date": trade_dt,
                    "order_book_id": oid,
                    "underlying": float(spot),
                    "S0": float(spot),
                    "r": float(risk_free_rate),
                    "q": float(dividend_yield),
                    "maturity": float(maturity),
                    "strike": float(strike),
                    "option_type": cp,
                    "market_price": float(market_price),
                    "volume": float(volume),
                    "days_to_expiry": days,
                }
            )
        if not output:
            empty = pd.DataFrame()
            return (empty, "all_contracts_filtered_out") if return_reason else empty
        df_out = pd.DataFrame(output).sort_values(["trade_date", "maturity", "strike"]).reset_index(drop=True)
        if include_api_iv:
            df_api_iv = self.get_option_api_iv(
                order_book_ids=df_out["order_book_id"].astype(str).tolist(),
                date=trade_date,
                model=greeks_model,
                price_type=greeks_price_type,
            )
            if not df_api_iv.empty:
                df_out = df_out.merge(df_api_iv, on="order_book_id", how="left")
            else:
                df_out["api_iv"] = np.nan
        df_out = self._align_option_panel_columns(df_out)
        return (df_out, "ok") if return_reason else df_out

    def clean_option_cross_section(
        self,
        df: pd.DataFrame,
        min_days_to_expiry: int = 1,
        min_volume: float = 0.0,
        max_abs_log_moneyness: Optional[float] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=self._CANONICAL_COLUMNS + self._EXTRA_COLUMNS)
        out = self._align_option_panel_columns(df)
        out = out.dropna(subset=["trade_date", "order_book_id", "maturity", "strike", "option_type", "market_price", "underlying", "r", "q"])
        out = out[(out["market_price"] > 0) & (out["maturity"] > 0) & (out["strike"] > 0)]
        out = out[out["option_type"].isin(["call", "put"])]
        if "days_to_expiry" in out.columns:
            out = out[out["days_to_expiry"] >= int(min_days_to_expiry)]
        if "volume" in out.columns:
            out = out[out["volume"] >= float(min_volume)]
        if max_abs_log_moneyness is not None:
            forward = out["underlying"] * np.exp((out["r"] - out["q"]) * out["maturity"])
            out["log_moneyness"] = np.log(out["strike"] / forward)
            out = out[np.abs(out["log_moneyness"]) <= float(max_abs_log_moneyness)]
        out = out.drop_duplicates(subset=["trade_date", "order_book_id"], keep="last")
        out = out.sort_values(["trade_date", "maturity", "strike"]).reset_index(drop=True)
        return self._align_option_panel_columns(out)

    @staticmethod
    def save_option_panel_csv(df: pd.DataFrame, save_path: str) -> str:
        if df is None or df.empty:
            raise ValueError("待保存数据为空，未写入文件。")
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        if "trade_date" in out.columns:
            out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
        out.to_csv(p, index=False, encoding="utf-8")
        return str(p)

    def fetch_option_panel_for_underlying(
        self,
        underlying_order_book_id: str,
        start_date: str,
        end_date: str,
        risk_free_rate: float = 0.02,
        dividend_yield: float = 0.0,
        apply_cleaning: bool = True,
        save_path: Optional[str] = None,
        include_api_iv: bool = False,
        verbose: bool = True,
        min_days_to_expiry: int = 5,
        min_price: float = 1e-4,
        min_volume: float = 0.0,
        max_abs_log_moneyness: Optional[float] = None,
        contract_snapshot_step: int = 5,
    ) -> pd.DataFrame:
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        if verbose:
            print("[OptionDataFetcher] step1/4 获取时间段合约池")

        trading_days = self.get_trading_days(start_date=start_date, end_date=end_date)
        if not trading_days:
            return pd.DataFrame(columns=self._CANONICAL_COLUMNS + self._EXTRA_COLUMNS)

        step = max(1, int(contract_snapshot_step))
        snapshot_days = trading_days[::step]
        if trading_days[-1] not in snapshot_days:
            snapshot_days.append(trading_days[-1])

        inst_chunks = []
        for d in snapshot_days:
            df_d = rq.all_instruments(type="Option", date=d)
            if df_d is None or len(df_d) == 0:
                continue
            df_d = df_d.copy()
            if any(c not in df_d.columns for c in self._INSTRUMENT_REQUIRED):
                continue
            df_d = df_d[df_d["underlying_order_book_id"].astype(str) == str(underlying_order_book_id)].copy()
            if not df_d.empty:
                inst_chunks.append(df_d)

        if not inst_chunks:
            return pd.DataFrame(columns=self._CANONICAL_COLUMNS + self._EXTRA_COLUMNS)

        df_inst = pd.concat(inst_chunks, axis=0, ignore_index=True)
        df_inst = df_inst.drop_duplicates(subset=["order_book_id"], keep="last").reset_index(drop=True)

        sdt = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
        edt = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
        inst = df_inst.copy()
        inst["listed_date"] = pd.to_datetime(inst["listed_date"], errors="coerce")
        inst["de_listed_date"] = pd.to_datetime(inst["de_listed_date"], errors="coerce")
        inst["maturity_date"] = pd.to_datetime(inst["maturity_date"], errors="coerce")
        inst = inst[(inst["listed_date"].isna()) | (inst["listed_date"] <= edt)]
        inst = inst[(inst["de_listed_date"].isna()) | (inst["de_listed_date"] >= sdt)]
        if inst.empty:
            return pd.DataFrame(columns=self._CANONICAL_COLUMNS + self._EXTRA_COLUMNS)

        option_ids = inst["order_book_id"].astype(str).drop_duplicates().tolist()

        if verbose:
            print(
                f"[OptionDataFetcher] step2/4 批量拉取期权价格: {len(option_ids)} 个合约 "
                f"(snapshot_days={len(snapshot_days)}, step={step})"
            )
        px = rq.get_price(
            order_book_ids=option_ids,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=["open", "close", "volume", "open_interest"],
            adjust_type="none",
            expect_df=True,
        )
        if px is None or len(px) == 0:
            return pd.DataFrame(columns=self._CANONICAL_COLUMNS + self._EXTRA_COLUMNS)
        px = px.reset_index()
        px["order_book_id"] = px["order_book_id"].astype(str)
        px["date"] = pd.to_datetime(px["date"], errors="coerce")
        px["market_price"] = 0.5 * (pd.to_numeric(px["open"], errors="coerce") + pd.to_numeric(px["close"], errors="coerce"))
        px["market_price"] = px["market_price"].where(~px["market_price"].isna(), pd.to_numeric(px["close"], errors="coerce"))
        px["volume"] = pd.to_numeric(px["volume"], errors="coerce").fillna(0.0)

        if verbose:
            print("[OptionDataFetcher] step3/4 拉取标的数据并合并")
        spot = self.get_underlying_daily_price(
            symbol=underlying_order_book_id,
            start_date=start_date,
            end_date=end_date,
            fields=["close", "open"],
        ).reset_index()
        spot["date"] = pd.to_datetime(spot["date"], errors="coerce")
        spot["underlying"] = pd.to_numeric(spot["close"], errors="coerce")
        spot["underlying"] = spot["underlying"].where(~spot["underlying"].isna(), pd.to_numeric(spot["open"], errors="coerce"))
        spot = spot[["date", "underlying"]]

        inst_small = inst[["order_book_id", "option_type", "strike_price", "maturity_date"]].copy()
        inst_small["order_book_id"] = inst_small["order_book_id"].astype(str)

        out = px.merge(inst_small, on="order_book_id", how="left")
        out = out.merge(spot, on="date", how="left")
        out["trade_date"] = out["date"]
        out["strike"] = pd.to_numeric(out["strike_price"], errors="coerce")
        out["option_type"] = out["option_type"].map(self._normalize_option_type)
        out["days_to_expiry"] = (pd.to_datetime(out["maturity_date"], errors="coerce") - out["trade_date"]).dt.days
        out["maturity"] = out["days_to_expiry"] / 365.0
        out["r"] = float(risk_free_rate)
        out["q"] = float(dividend_yield)
        out["S0"] = out["underlying"]

        out = out[
            (out["market_price"] >= float(min_price))
            & (out["volume"] >= float(min_volume))
            & (out["days_to_expiry"] >= int(min_days_to_expiry))
        ]
        out = self._align_option_panel_columns(out)

        if apply_cleaning and not out.empty:
            out = self.clean_option_cross_section(
                out,
                min_days_to_expiry=min_days_to_expiry,
                min_volume=min_volume,
                max_abs_log_moneyness=max_abs_log_moneyness,
            )

        if save_path is not None and not out.empty:
            saved = self.save_option_panel_csv(out, save_path)
            if verbose:
                print(f"✓ 已保存期权面板数据: {saved}")

        if verbose:
            print(f"✓ 期权面板完成: {len(out)} 行, 覆盖 {out['trade_date'].nunique() if 'trade_date' in out.columns else 0} 个交易日")
        return out
