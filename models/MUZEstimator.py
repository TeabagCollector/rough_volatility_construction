"""
MUZ波动率估计模块
基于Robert & Rosenbaum (2011)的Model with Uncertainty Zones实现
从高频数据中提取瞬时方差代理值
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from DataFetcher import DataFetcher


class MUZEstimator:
    """
    MUZ (Model with Uncertainty Zones) 估计器
    
    用于从高频tick数据中估计日度瞬时方差代理值
    """
    
    def __init__(self, data_fetcher: Optional[DataFetcher] = None):
        """
        初始化MUZ估计器
        
        Args:
            data_fetcher: DataFetcher实例，如果不提供则创建新实例
        """
        self.data_fetcher = data_fetcher if data_fetcher else DataFetcher()
    
    def _calculate_eta(
        self, 
        prices: pd.Series, 
        tick_size: float
    ) -> Tuple[float, int, int]:
        """
        计算价格变动厌恶参数 η
        
        基于累积价格变动达到tick_size时记录跳变，统计交替和连续模式
        
        算法逻辑：
        1. 累积价格变动，直到|累积变动| >= tick_size
        2. 达到阈值时记录一次有效跳变及其方向
        3. 重置累积变动，继续下一轮累积
        4. 统计连续跳变间的方向关系（alternation vs continuation）
        
        Args:
            prices: 价格序列
            tick_size: tick大小
        
        Returns:
            (eta, N_alternation, N_continuation): η值、交替次数、连续次数
        """
        if len(prices) < 2:
            return np.nan, 0, 0
        
        cumulative_change = 0.0
        last_direction = None
        N_alternation = 0
        N_continuation = 0
        
        for i in range(1, len(prices)):
            cumulative_change += prices.iloc[i] - prices.iloc[i-1]
            
            # 检查累积变动是否达到±1个tick
            if abs(cumulative_change) >= tick_size:
                # 确定本次跳变方向
                current_direction = np.sign(cumulative_change)
                
                # 判断与上次跳变的关系
                if last_direction is not None:
                    if current_direction == last_direction:
                        N_continuation += 1
                    else:
                        N_alternation += 1
                
                # 更新状态并重置累积
                last_direction = current_direction
                cumulative_change = 0.0
        
        # 计算 eta
        if N_alternation == 0:
            eta = np.nan
        else:
            eta = N_continuation / (2 * N_alternation)
        
        return eta, N_alternation, N_continuation
    
    def find_optimal_tick_size(
        self,
        prices: pd.Series,
        candidates: Optional[List[float]] = None,
        target_eta: float = 0.5,
        verbose: bool = False
    ) -> Tuple[float, float, Dict]:
        """
        寻找最优的tick size使得η接近目标值(默认0.5)
        
        Args:
            prices: 价格序列
            candidates: 候选tick size列表，如果不提供则自动生成
            target_eta: 目标η值，默认0.5
            verbose: 是否显示详细信息
        
        Returns:
            (optimal_tick_size, optimal_eta, results_dict): 最优tick、对应η值、所有结果
        """
        if candidates is None:
            price_range = prices.max() - prices.min()
            mean_price = prices.mean()
            
            candidates = [
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.015,
                0.02,
                0.025,
                0.03,
                0.001 * mean_price,
                0.002 * mean_price,
                0.005 * mean_price,
                0.01 * mean_price,
                0.02 * mean_price,
                0.05 * mean_price,
                0.1 * mean_price
            ]
        
        results = {}
        min_distance = float('inf')
        optimal_tick = candidates[0]
        optimal_eta = np.nan
        
        for tick in candidates:
            eta, n_alt, n_cont = self._calculate_eta(prices, tick)
            results[tick] = {
                'eta': eta,
                'N_alternation': n_alt,
                'N_continuation': n_cont,
                'distance_to_target': abs(eta - target_eta) if not np.isnan(eta) else float('inf')
            }
            
            if not np.isnan(eta):
                distance = abs(eta - target_eta)
                if distance < min_distance:
                    min_distance = distance
                    optimal_tick = tick
                    optimal_eta = eta
        
        if verbose:
            print(f"\n=== Tick Size 优化结果 ===")
            print(f"目标 η: {target_eta}")
            print(f"\n候选结果:")
            for tick, res in sorted(results.items()):
                if not np.isnan(res['eta']):
                    print(f"  tick={tick:.6f}: η={res['eta']:.4f}, "
                          f"N_alt={res['N_alternation']}, N_cont={res['N_continuation']}, "
                          f"距离={res['distance_to_target']:.4f}")
            print(f"\n✓ 最优 tick size: {optimal_tick:.6f}")
            print(f"  对应 η: {optimal_eta:.4f}")
        
        return optimal_tick, optimal_eta, results
    
    def _reconstruct_efficient_price(
        self,
        prices: pd.Series,
        eta: float,
        tick_size: float
    ) -> pd.Series:
        """
        重构有效价格（去除买卖价差噪声）
        
        基于累积跳变的有效价格重构：
        1. 累积价格变动直到达到tick_size
        2. 只对有效跳变应用修正：correction = sign * (0.5 - eta) * tick_size
        3. 小于tick_size的噪音变动被自然过滤掉
        
        Args:
            prices: 观测价格序列
            eta: 价格变动厌恶参数
            tick_size: tick大小
        
        Returns:
            重构后的有效价格序列
        """
        efficient_prices = pd.Series(index=prices.index, dtype=float)
        efficient_prices.iloc[0] = prices.iloc[0]
        
        cumulative_change = 0.0
        last_efficient_price = prices.iloc[0]
        
        for i in range(1, len(prices)):
            cumulative_change += prices.iloc[i] - prices.iloc[i-1]
            
            # 只有达到tick_size的跳变才修正
            if abs(cumulative_change) >= tick_size:
                direction = np.sign(cumulative_change)
                correction = direction * (0.5 - eta) * tick_size
                
                # 更新有效价格：累积变动 - 买卖价差修正
                last_efficient_price += (cumulative_change - correction)
                cumulative_change = 0.0
            
            efficient_prices.iloc[i] = last_efficient_price
        
        return efficient_prices
    
    def calculate_rv_uz(
        self,
        prices: pd.Series,
        tick_size: Optional[float] = None,
        optimize_tick: bool = True,
        verbose: bool = False
    ) -> Tuple[float, Dict]:
        """
        计算基于MUZ模型的实现方差 (RV^UZ)
        
        Args:
            prices: 价格序列
            tick_size: tick大小，如果optimize_tick=True则会被优化
            optimize_tick: 是否优化tick size
            verbose: 是否显示详细信息
        
        Returns:
            (rv_uz, info_dict): 实现方差值和计算信息
        """
        if len(prices) < 2:
            return np.nan, {'error': 'insufficient data'}
        
        if optimize_tick or tick_size is None:
            optimal_tick, optimal_eta, opt_results = self.find_optimal_tick_size(
                prices, verbose=verbose
            )
            tick_size = optimal_tick
            eta = optimal_eta
        else:
            eta, n_alt, n_cont = self._calculate_eta(prices, tick_size)
        
        if np.isnan(eta):
            return np.nan, {'error': 'failed to estimate eta'}
        
        efficient_prices = self._reconstruct_efficient_price(prices, eta, tick_size)
        
        log_returns = np.log(efficient_prices / efficient_prices.shift(1)).dropna()
        rv_uz = np.sum(log_returns ** 2)
        
        info = {
            'tick_size': tick_size,
            'eta': eta,
            'n_observations': len(prices),
            'n_returns': len(log_returns),
            'rv_uz': rv_uz
        }
        
        return rv_uz, info
    
    def process_single_day(
        self,
        symbol: str,
        date: str,
        tick_size: Optional[float] = None,
        optimize_tick: bool = True,
        verbose: bool = False
    ) -> Tuple[Optional[float], Dict]:
        """
        处理单个交易日的数据
        
        Args:
            symbol: 标的代码
            date: 交易日期 'YYYYMMDD'
            tick_size: tick大小
            optimize_tick: 是否优化tick size
            verbose: 是否显示详细信息
        
        Returns:
            (rv_uz, info): 该日的方差代理值和信息
        """
        df = self.data_fetcher.get_tick_data(symbol, date)
        
        if df.empty:
            return None, {'error': 'no data', 'date': date}
        
        prices = df['last']
        
        rv_uz, info = self.calculate_rv_uz(
            prices, 
            tick_size=tick_size,
            optimize_tick=optimize_tick,
            verbose=verbose
        )
        
        info['date'] = date
        info['data_points'] = len(df)
        
        return rv_uz, info
    
    def process_period(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        tick_size: Optional[float] = None,
        optimize_tick: bool = True,
        adaptive_tick: bool = False,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        处理整个时间段的数据
        
        Args:
            symbol: 标的代码
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
            tick_size: 初始tick大小
            optimize_tick: 是否优化tick size
            adaptive_tick: 是否对每个交易日自适应优化tick（计算较慢）
            verbose: 是否显示进度信息
        
        Returns:
            包含日期、方差代理值和元信息的DataFrame
        """
        if not self.data_fetcher.is_connected:
            raise RuntimeError("DataFetcher未连接，请先调用init_connection()")
        
        trading_days = self.data_fetcher.get_trading_days(start_date, end_date)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"开始处理 {len(trading_days)} 个交易日的数据")
            print(f"标的: {symbol}")
            print(f"时间范围: {start_date} - {end_date}")
            print(f"{'='*60}\n")
        
        results = []
        failed_count = 0
        
        for i, day in enumerate(trading_days):
            if verbose and (i % 50 == 0 or i == len(trading_days) - 1):
                print(f"进度: {i+1}/{len(trading_days)} ({100*(i+1)/len(trading_days):.1f}%)")
            
            rv_uz, info = self.process_single_day(
                symbol=symbol,
                date=day,
                tick_size=tick_size,
                optimize_tick=optimize_tick if adaptive_tick else (i == 0 and optimize_tick),
                verbose=False
            )
            
            if rv_uz is not None and not np.isnan(rv_uz):
                results.append({
                    'date': day,
                    'rv_uz': rv_uz,
                    'tick_size': info.get('tick_size'),
                    'eta': info.get('eta'),
                    'n_observations': info.get('data_points', 0)
                })
                
                if not adaptive_tick and i == 0 and 'tick_size' in info:
                    tick_size = info['tick_size']
                    if verbose:
                        print(f"✓ 首日优化完成，后续使用 tick_size = {tick_size:.6f}")
            else:
                failed_count += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ 处理完成!")
            print(f"  成功: {len(results)}/{len(trading_days)} 个交易日")
            print(f"  失败: {failed_count} 个交易日")
            print(f"{'='*60}\n")
        
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
            df_results = df_results.set_index('date').sort_index()
        
        return df_results

