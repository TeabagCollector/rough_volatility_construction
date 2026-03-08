"""
MUZ (Model with Uncertainty Zones) volatility estimation module.
Based on Robert & Rosenbaum (2011), extracts instantaneous variance proxy from high-frequency data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from DataFetcher import DataFetcher


class MUZEstimator:
    """
    MUZ (Model with Uncertainty Zones) estimator.
    
    Estimates daily instantaneous variance proxy from high-frequency tick data.
    """
    
    def __init__(self, data_fetcher: Optional[DataFetcher] = None):
        """
        Initialize MUZ estimator.
        
        Args:
            data_fetcher: DataFetcher instance. Creates new one if not provided.
        """
        self.data_fetcher = data_fetcher if data_fetcher else DataFetcher()
    
    def _calculate_eta(
        self, 
        prices: pd.Series, 
        tick_size: float
    ) -> Tuple[float, int, int]:
        """
        Compute price change aversion parameter eta.
        
        Records jumps when cumulative price change reaches tick_size, counts alternation vs continuation.
        
        Algorithm:
        1. Accumulate price changes until |cumulative| >= tick_size
        2. Record valid jump and direction at threshold
        3. Reset cumulative, continue
        4. Count direction relationship between consecutive jumps (alternation vs continuation)
        
        Args:
            prices: Price series
            tick_size: Tick size
        
        Returns:
            (eta, N_alternation, N_continuation)
        """
        if len(prices) < 2:
            return np.nan, 0, 0
        
        cumulative_change = 0.0
        last_direction = None
        N_alternation = 0
        N_continuation = 0
        
        for i in range(1, len(prices)):
            cumulative_change += prices.iloc[i] - prices.iloc[i-1]
            
            if abs(cumulative_change) >= tick_size:
                current_direction = np.sign(cumulative_change)
                
                if last_direction is not None:
                    if current_direction == last_direction:
                        N_continuation += 1
                    else:
                        N_alternation += 1
                
                last_direction = current_direction
                cumulative_change = 0.0
        
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
        Find optimal tick size such that eta is close to target (default 0.5).
        
        Args:
            prices: Price series
            candidates: Candidate tick sizes. Auto-generated if not provided.
            target_eta: Target eta value, default 0.5
            verbose: Whether to print details
        
        Returns:
            (optimal_tick_size, optimal_eta, results_dict)
        """
        if candidates is None:
            mean_price = prices.mean()
            candidates = [
                0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
                0.015, 0.02, 0.025, 0.03,
                0.001 * mean_price, 0.002 * mean_price, 0.005 * mean_price,
                0.01 * mean_price, 0.02 * mean_price, 0.05 * mean_price, 0.1 * mean_price
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
            print(f"\n=== Tick Size Optimization ===")
            print(f"Target eta: {target_eta}")
            print(f"\nCandidates:")
            for tick, res in sorted(results.items()):
                if not np.isnan(res['eta']):
                    print(f"  tick={tick:.6f}: eta={res['eta']:.4f}, "
                          f"N_alt={res['N_alternation']}, N_cont={res['N_continuation']}, "
                          f"distance={res['distance_to_target']:.4f}")
            print(f"\nOptimal tick size: {optimal_tick:.6f}")
            print(f"Corresponding eta: {optimal_eta:.4f}")
        
        return optimal_tick, optimal_eta, results
    
    def _reconstruct_efficient_price(
        self,
        prices: pd.Series,
        eta: float,
        tick_size: float
    ) -> pd.Series:
        """
        Reconstruct efficient price (remove bid-ask spread noise).
        
        Based on cumulative jump correction:
        1. Accumulate until tick_size
        2. Apply correction for valid jumps: correction = sign * (0.5 - eta) * tick_size
        3. Noise below tick_size is naturally filtered
        
        Args:
            prices: Observed price series
            eta: Price change aversion parameter
            tick_size: Tick size
        
        Returns:
            Reconstructed efficient price series
        """
        efficient_prices = pd.Series(index=prices.index, dtype=float)
        efficient_prices.iloc[0] = prices.iloc[0]
        
        cumulative_change = 0.0
        last_efficient_price = prices.iloc[0]
        
        for i in range(1, len(prices)):
            cumulative_change += prices.iloc[i] - prices.iloc[i-1]
            
            if abs(cumulative_change) >= tick_size:
                direction = np.sign(cumulative_change)
                correction = direction * (0.5 - eta) * tick_size
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
        Compute realized variance (RV^UZ) based on MUZ model.
        
        Args:
            prices: Price series
            tick_size: Tick size (optimized if optimize_tick=True)
            optimize_tick: Whether to optimize tick size
            verbose: Whether to print details
        
        Returns:
            (rv_uz, info_dict)
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
        Process a single trading day.
        
        Args:
            symbol: Symbol code
            date: Trading date 'YYYYMMDD'
            tick_size: Tick size
            optimize_tick: Whether to optimize tick size
            verbose: Whether to print details
        
        Returns:
            (rv_uz, info)
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
        Process data over a date range.
        
        Args:
            symbol: Symbol code
            start_date: Start date 'YYYYMMDD'
            end_date: End date 'YYYYMMDD'
            tick_size: Initial tick size
            optimize_tick: Whether to optimize tick size
            adaptive_tick: Whether to optimize tick per day (slower)
            verbose: Whether to print progress
        
        Returns:
            DataFrame with date, variance proxy, and metadata
        """
        if not self.data_fetcher.is_connected:
            raise RuntimeError("DataFetcher not connected. Call init_connection() first.")
        
        trading_days = self.data_fetcher.get_trading_days(start_date, end_date)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {len(trading_days)} trading days")
            print(f"Symbol: {symbol}")
            print(f"Range: {start_date} - {end_date}")
            print(f"{'='*60}\n")
        
        results = []
        failed_count = 0
        
        for i, day in enumerate(trading_days):
            if verbose and (i % 50 == 0 or i == len(trading_days) - 1):
                print(f"Progress: {i+1}/{len(trading_days)} ({100*(i+1)/len(trading_days):.1f}%)")
            
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
                        print(f"First-day optimization done, using tick_size = {tick_size:.6f}")
            else:
                failed_count += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Done!")
            print(f"  Success: {len(results)}/{len(trading_days)} days")
            print(f"  Failed: {failed_count} days")
            print(f"{'='*60}\n")
        
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
            df_results = df_results.set_index('date').sort_index()
        
        return df_results
