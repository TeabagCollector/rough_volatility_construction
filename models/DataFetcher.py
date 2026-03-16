"""
期权数据获取模块
通过米筐API获取期权完整数据和Greeks信息
"""

import os
import rqdatac as rq
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict, Tuple
import datetime
import warnings
warnings.filterwarnings('ignore')

# 米筐 API license 从环境变量 RQDATAC_LICENSE 读取，避免硬编码敏感信息
# 使用前请设置: export RQDATAC_LICENSE='your_license_key'


class UnderlyingDataFetcher():
    
    def __init__(self, license: Optional[str] = None):
        """
        初始化数据获取器
        
        Args:
            license: 米筐API的license key，如果不提供则需要后续调用init_connection时提供
            data_dir: 数据保存目录，默认为当前目录下的data文件夹
        """
        self.license = license
        self.is_connected = False

    def init_connection(self, license: Optional[str] = None):
        """
        初始化米筐数据连接
        
        Args:
            license: 米筐API的license key，若不提供则从环境变量 RQDATAC_LICENSE 读取
        """
        if license:
            self.license = license
        elif not self.license:
            self.license = os.environ.get('RQDATAC_LICENSE')
            
        if not self.license:
            raise ValueError(
                "请提供米筐API的license key。可通过 init_connection(license='...') 传入，"
                "或设置环境变量: export RQDATAC_LICENSE='your_license_key'"
            )
            
        try:
            rq.init('license', self.license)
            self.is_connected = True
            print("✓ 米筐数据连接成功")
        except Exception as e:
            print(f"✗ 米筐数据连接失败: {e}")
            raise
    
    def get_trading_days(self, start_date: str, end_date: str, market: str = 'cn') -> List[str]:
        """
        获取指定时间段内的交易日
        
        Args:
            start_date: 开始日期，格式'YYYYMMDD'
            end_date: 结束日期，格式'YYYYMMDD'
            market: 市场代码，默认为'cn'（中国市场）
        
        Returns:
            交易日列表
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        
        try:
            trading_days = rq.get_trading_dates(start_date, end_date, market=market)
            trading_days_str = [day.strftime('%Y%m%d') for day in trading_days]
            return trading_days_str
        except Exception as e:
            print(f"✗ 获取交易日失败: {e}")
            raise
    
    def get_tick_data(
        self, 
        symbol: str, 
        date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0)
    ) -> pd.DataFrame:
        """
        获取指定标的在特定日期和时间窗口内的tick数据
        
        Args:
            symbol: 标的代码，如'000300.XSHG'
            date: 交易日期，格式'YYYYMMDD'
            start_time: 时间窗口开始时间
            end_time: 时间窗口结束时间
        
        Returns:
            包含tick数据的DataFrame
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")
        
        try:
            df = rq.get_price(
                symbol,
                start_date=date,
                end_date=date,
                frequency='tick',
                time_slice=(start_time, end_time)
            )
            return df
        except Exception as e:
            print(f"✗ 获取{date}的tick数据失败: {e}")
            return pd.DataFrame()
    
    def get_tick_data_batch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0),
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        批量获取指定时间段内的tick数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期，格式'YYYYMMDD'
            end_date: 结束日期，格式'YYYYMMDD'
            start_time: 每日时间窗口开始时间
            end_time: 每日时间窗口结束时间
            verbose: 是否显示进度信息
        
        Returns:
            合并后的DataFrame
        """
        trading_days = self.get_trading_days(start_date, end_date)
        
        all_data = []
        failed_days = []
        
        for i, day in enumerate(trading_days):
            if verbose and (i % 10 == 0 or i == len(trading_days) - 1):
                print(f"正在获取数据: {i+1}/{len(trading_days)} - {day}")
            
            df = self.get_tick_data(symbol, day, start_time, end_time)
            
            if not df.empty:
                all_data.append(df)
            else:
                failed_days.append(day)
        
        if verbose:
            print(f"✓ 成功获取 {len(all_data)}/{len(trading_days)} 个交易日的数据")
            if failed_days:
                print(f"⚠ 失败的日期: {failed_days}")
        
        if all_data:
            return pd.concat(all_data, axis=0)
        else:
            return pd.DataFrame()

    def get_minute_data(
        self,
        symbol: str,
        date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0),
        frequency: str = '1m',
        fields: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        获取指定标的在特定日期和时间窗口内的分钟级数据。

        说明：
        - 基于 rq.get_price(..., frequency='1m', time_slice=(start_time, end_time))
        - 默认口径沿用当前 tick 主干（10:00-11:00），但保留参数可配置
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")

        try:
            df = rq.get_price(
                symbol,
                start_date=date,
                end_date=date,
                frequency=frequency,
                fields=fields,
                time_slice=(start_time, end_time)
            )
            if isinstance(df, pd.Series):
                col_name = fields if isinstance(fields, str) else 'value'
                df = df.to_frame(name=col_name)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            print(f"✗ 获取{date}的分钟数据失败: {e}")
            return pd.DataFrame()

    def get_minute_data_batch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0),
        frequency: str = '1m',
        fields: Optional[Union[str, List[str]]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        批量获取指定时间段内的分钟级数据。
        """
        trading_days = self.get_trading_days(start_date, end_date)

        all_data = []
        failed_days = []

        for i, day in enumerate(trading_days):
            if verbose and (i % 10 == 0 or i == len(trading_days) - 1):
                print(f"正在获取分钟数据: {i+1}/{len(trading_days)} - {day}")

            df = self.get_minute_data(
                symbol=symbol,
                date=day,
                start_time=start_time,
                end_time=end_time,
                frequency=frequency,
                fields=fields
            )
            if not df.empty:
                all_data.append(df)
            else:
                failed_days.append(day)

        if verbose:
            print(f"✓ 成功获取 {len(all_data)}/{len(trading_days)} 个交易日的分钟数据")
            if failed_days:
                print(f"⚠ 失败的日期: {failed_days}")

        if all_data:
            return pd.concat(all_data, axis=0)
        return pd.DataFrame()

    @staticmethod
    def _extract_datetime_index(df: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
        """
        尝试从 DataFrame 中提取 datetime 索引（兼容普通索引、MultiIndex、datetime 列）。
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        if isinstance(df.index, pd.DatetimeIndex):
            return df.index

        if isinstance(df.index, pd.MultiIndex):
            index_names = [name for name in df.index.names if name is not None]
            if 'datetime' in index_names:
                dt_idx = df.index.get_level_values('datetime')
                return pd.to_datetime(dt_idx, errors='coerce')

            for level in range(df.index.nlevels):
                candidate = pd.to_datetime(df.index.get_level_values(level), errors='coerce')
                if isinstance(candidate, pd.DatetimeIndex) and candidate.notna().sum() >= max(1, len(df) // 2):
                    return candidate

        if 'datetime' in df.columns:
            dt_col = pd.to_datetime(df['datetime'], errors='coerce')
            if dt_col.notna().any():
                return pd.DatetimeIndex(dt_col)

        return None

    def _get_intraday_price_series(
        self,
        df: pd.DataFrame,
        price_col: Optional[str] = None
    ) -> pd.Series:
        """
        从日内数据中提取价格序列（优先 close，其次 last/open）。
        """
        if df.empty:
            return pd.Series(dtype=float)

        col = price_col if price_col in df.columns else self._pick_existing_column(
            df, ['close', 'last', 'open', 'settlement']
        )
        if col is None:
            return pd.Series(dtype=float)

        prices = pd.to_numeric(df[col], errors='coerce')
        dt_idx = self._extract_datetime_index(df)
        if dt_idx is not None and len(dt_idx) == len(prices):
            prices = pd.Series(prices.values, index=dt_idx)
        else:
            prices = pd.Series(prices.values)
        prices = prices.dropna()
        if isinstance(prices.index, pd.DatetimeIndex):
            prices = prices[~prices.index.duplicated(keep='last')].sort_index()
        return prices

    @staticmethod
    def calculate_realized_variance(
        prices: pd.Series,
        use_log_return: bool = True,
        min_obs: int = 5
    ) -> Tuple[Optional[float], Dict]:
        """
        基于日内价格序列计算实现方差：
            RV = Σ r_t^2
        """
        if prices is None or len(prices) < max(2, min_obs):
            return None, {'error': 'insufficient observations'}

        prices = pd.to_numeric(prices, errors='coerce').dropna()
        prices = prices[prices > 0]
        if len(prices) < max(2, min_obs):
            return None, {'error': 'insufficient positive prices'}

        if use_log_return:
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()

        if len(returns) < max(1, min_obs - 1):
            return None, {'error': 'insufficient returns'}

        rv = float(np.sum(np.square(returns.values)))
        if not np.isfinite(rv):
            return None, {'error': 'rv is not finite'}

        info = {
            'n_observations': int(len(prices)),
            'n_returns': int(len(returns)),
            'use_log_return': bool(use_log_return),
            'rv': rv
        }
        return rv, info

    def process_minute_variance_period(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        start_time: datetime.time = datetime.time(hour=10, minute=0),
        end_time: datetime.time = datetime.time(hour=11, minute=0),
        frequency: str = '1m',
        fields: Optional[Union[str, List[str]]] = None,
        price_col: Optional[str] = None,
        use_log_return: bool = True,
        min_obs: int = 5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        用分钟级价格构建日度代理方差（不使用 tick / MUZ）。
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")

        trading_days = self.get_trading_days(start_date, end_date)
        results = []
        failed_days = []

        for i, day in enumerate(trading_days):
            if verbose and (i % 20 == 0 or i == len(trading_days) - 1):
                print(f"分钟代理方差计算进度: {i+1}/{len(trading_days)} - {day}")

            df = self.get_minute_data(
                symbol=symbol,
                date=day,
                start_time=start_time,
                end_time=end_time,
                frequency=frequency,
                fields=fields
            )
            if df.empty:
                failed_days.append((day, 'no data'))
                continue

            prices = self._get_intraday_price_series(df, price_col=price_col)
            if prices.empty:
                failed_days.append((day, 'no valid price column'))
                continue

            rv, info = self.calculate_realized_variance(
                prices=prices,
                use_log_return=use_log_return,
                min_obs=min_obs
            )
            if rv is None:
                failed_days.append((day, info.get('error', 'rv failed')))
                continue

            results.append({
                'date': pd.to_datetime(day, format='%Y%m%d'),
                'rv_minute': rv,
                'n_observations': info.get('n_observations', len(prices)),
                'n_returns': info.get('n_returns', max(0, len(prices) - 1))
            })

        if not results:
            if verbose:
                print("⚠ 未生成有效分钟代理方差结果")
            return pd.DataFrame()

        df_out = pd.DataFrame(results).set_index('date').sort_index()
        df_out['annualized_vol'] = np.sqrt(df_out['rv_minute'] * 252.0) * 100.0

        if verbose:
            print(f"✓ 分钟代理方差计算完成: {len(df_out)}/{len(trading_days)} 个交易日有效")
            if failed_days:
                preview = failed_days[:10]
                print(f"⚠ 失败样本（前10条）: {preview}")
        return df_out

    @staticmethod
    def _pick_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """从候选列名中返回第一个在 DataFrame 中存在的列名。"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _normalize_option_type(value: Union[str, int, float]) -> Optional[str]:
        """
        统一期权类型字段到 {'call','put'}。
        支持 C/P、CALL/PUT、认购/认沽、1/-1 等常见编码。
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None

        if isinstance(value, (int, np.integer)):
            if int(value) == 1:
                return 'call'
            if int(value) == -1:
                return 'put'

        text = str(value).strip().lower()
        if text in {'c', 'call', '认购', '购'}:
            return 'call'
        if text in {'p', 'put', '认沽', '沽'}:
            return 'put'
        return None

    def get_underlying_daily_price(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取标的日频价格序列。
        """
        if not self.is_connected:
            raise RuntimeError("请先调用init_connection()初始化连接")

        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume']

        try:
            df = rq.get_price(
                symbol,
                start_date=start_date,
                end_date=end_date,
                frequency='1d',
                fields=fields
            )
            if isinstance(df, pd.Series):
                df = df.to_frame(name=fields[0] if fields else 'value')
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as e:
            print(f"✗ 获取标的日频数据失败: {e}")
            return pd.DataFrame()

    def get_underlying_spot_on_date(
        self,
        symbol: str,
        date: str
    ) -> Optional[float]:
        """
        获取标的在指定交易日的现价（优先 close，其次 open/high/low）。
        """
        df = self.get_underlying_daily_price(symbol, date, date)
        if df.empty:
            return None

        price_col = self._pick_existing_column(df, ['close', 'last', 'settlement', 'open'])
        if price_col is None:
            return None

        value = df[price_col].dropna()
        if value.empty:
            return None
        return float(value.iloc[-1])

        return float(value.iloc[-1])


class DataFetcher(UnderlyingDataFetcher):
    """
    向后兼容层：
    - Underlying 相关能力继续来自 UnderlyingDataFetcher
    - 期权相关能力委托给 OptionDataFetcher
    """

    def _option_delegate(self):
        from OptionDataFetcher import OptionDataFetcher
        return OptionDataFetcher(license=self.license)

    @staticmethod
    def _warn_migrated(method_name: str):
        warnings.warn(
            f"DataFetcher.{method_name} 已迁移到 OptionDataFetcher，请改用 OptionDataFetcher.{method_name}。",
            DeprecationWarning,
            stacklevel=3,
        )

    def _call_option_delegate(self, method_name: str, *args, **kwargs):
        self._warn_migrated(method_name)
        delegate = self._option_delegate()
        if method_name in {
            "get_option_instruments",
            "get_option_prices",
            "get_option_api_iv",
            "build_option_cross_section",
            "fetch_option_panel_for_underlying",
        }:
            delegate.init_connection(self.license)
        return getattr(delegate, method_name)(*args, **kwargs)

    def get_option_instruments(self, *args, **kwargs):
        return self._call_option_delegate("get_option_instruments", *args, **kwargs)

    def get_option_prices(self, *args, **kwargs):
        return self._call_option_delegate("get_option_prices", *args, **kwargs)

    def get_option_api_iv(self, *args, **kwargs):
        return self._call_option_delegate("get_option_api_iv", *args, **kwargs)

    def build_option_cross_section(self, *args, **kwargs):
        return self._call_option_delegate("build_option_cross_section", *args, **kwargs)

    def clean_option_cross_section(self, *args, **kwargs):
        return self._call_option_delegate("clean_option_cross_section", *args, **kwargs)

    def save_option_panel_csv(self, *args, **kwargs):
        return self._call_option_delegate("save_option_panel_csv", *args, **kwargs)

    def fetch_option_panel_for_underlying(self, *args, **kwargs):
        return self._call_option_delegate("fetch_option_panel_for_underlying", *args, **kwargs)