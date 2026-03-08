"""
Hurst exponent estimation module.
Based on Gatheral et al. (2014) "Volatility is Rough".
Uses variogram analysis and q-th moment regression to estimate Hurst exponent of fractional Brownian motion.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class HurstEstimator:
    """
    Hurst exponent estimator.
    
    Based on Gatheral et al. (2014) "Volatility is Rough".
    Uses variogram analysis and q-th moment regression.
    
    Main methods:
    - estimate_hurst_variogram: Single-q variogram estimation
    - estimate_hurst_multiscale: Multi-q robust estimation
    - estimate_hurst_roughness_test: Rough volatility test
    - plot_variogram: Variogram visualization
    """
    
    def __init__(self):
        """Initialize Hurst estimator."""
        pass
    
    def _calculate_qth_moment(
        self,
        series: pd.Series,
        lag: int,
        q: float = 2.0
    ) -> float:
        """
        Compute q-th moment of increments: E[|X(t+τ) - X(t)|^q].
        
        Args:
            series: Time series
            lag: Lag τ
            q: Moment order (default 2 = variance)
        
        Returns:
            Estimated q-th moment
        """
        if len(series) <= lag:
            return np.nan
        
        increments = series.diff(lag).dropna()
        if len(increments) == 0:
            return np.nan
        
        return np.mean(np.abs(increments) ** q)
    
    def estimate_hurst_variogram(
        self,
        series: pd.Series,
        lags: Optional[List[int]] = None,
        q: float = 2.0,
        log_space: bool = True
    ) -> Tuple[float, Dict]:
        """
        Estimate Hurst exponent via variogram (structure function).
        
        Theory: E[|X(t+τ) - X(t)|^q] ~ τ^(qH)
        Log form: log(E[|ΔX_τ|^q]) = qH * log(τ) + const
        Slope = qH, so H = slope / q.
        
        Args:
            series: Time series (typically log-variance)
            lags: Lag list. Auto-generated if not provided.
            q: Moment order, q=2 for standard variogram
            log_space: Whether to sample lags in log space
        
        Returns:
            (H, info_dict)
        """
        if len(series) < 10:
            return np.nan, {'error': 'insufficient data'}
        
        if lags is None:
            max_lag = min(len(series) // 3, 250)
            if log_space:
                lags = np.unique(np.logspace(0, np.log10(max_lag), 50).astype(int))
            else:
                lags = np.arange(1, max_lag, max(1, max_lag // 50))
        
        moments = []
        valid_lags = []
        
        for lag in lags:
            moment = self._calculate_qth_moment(series, lag, q)
            if not np.isnan(moment) and moment > 0:
                moments.append(moment)
                valid_lags.append(lag)
        
        if len(valid_lags) < 3:
            return np.nan, {'error': 'insufficient valid lags'}
        
        log_lags = np.log(valid_lags)
        log_moments = np.log(moments)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_moments)
        H = slope / q
        
        info = {
            'H': H, 'slope': slope, 'intercept': intercept,
            'r_squared': r_value ** 2, 'p_value': p_value, 'std_err': std_err,
            'q': q, 'n_lags': len(valid_lags), 'lags': valid_lags, 'moments': moments,
            'log_lags': log_lags.tolist(), 'log_moments': log_moments.tolist()
        }
        return H, info
    
    def estimate_hurst_multiscale(
        self,
        series: pd.Series,
        q_values: Optional[List[float]] = None,
        lags: Optional[List[int]] = None
    ) -> Tuple[float, Dict]:
        """
        Multi-scale Hurst estimation using multiple q values.
        
        Args:
            series: Time series
            q_values: List of q values, default [1.0, 1.5, 2.0]
            lags: Lag list
        
        Returns:
            (H_mean, info_dict)
        """
        if q_values is None:
            q_values = [1.0, 1.5, 2.0]
        
        results = {}
        H_estimates = []
        
        for q in q_values:
            H, info = self.estimate_hurst_variogram(series, lags=lags, q=q)
            results[f'q={q}'] = info
            if not np.isnan(H):
                H_estimates.append(H)
        
        if len(H_estimates) == 0:
            return np.nan, {'error': 'all q values failed', 'results': results}
        
        summary = {
            'H_mean': np.mean(H_estimates),
            'H_std': np.std(H_estimates),
            'H_estimates': H_estimates,
            'q_values': q_values,
            'n_valid': len(H_estimates),
            'results_by_q': results
        }
        return summary['H_mean'], summary
    
    def estimate_hurst_roughness_test(
        self,
        variance_series: pd.Series,
        q: float = 2.0,
        significance_level: float = 0.05
    ) -> Dict:
        """
        Rough volatility test: check if H is significantly < 0.5.
        
        Args:
            variance_series: Variance series (log(RV) or RV)
            q: Moment order
            significance_level: Significance level
        
        Returns:
            Dict with test results
        """
        H, info = self.estimate_hurst_variogram(variance_series, q=q)
        
        if np.isnan(H):
            return {'error': 'failed to estimate H', 'info': info}
        
        is_rough = H < 0.5
        std_err = info.get('std_err', np.nan)
        if not np.isnan(std_err):
            H_std_err = std_err / q
            t_critical = stats.t.ppf(1 - significance_level/2, info['n_lags'] - 2)
            ci_lower = H - t_critical * H_std_err
            ci_upper = H + t_critical * H_std_err
        else:
            ci_lower = ci_upper = np.nan
        
        return {
            'H': H, 'is_rough': is_rough,
            'is_significant': ci_upper < 0.5,
            'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'r_squared': info.get('r_squared'), 'p_value': info.get('p_value'),
            'n_lags': info.get('n_lags'),
            'interpretation': self._interpret_hurst(H)
        }
    
    def _interpret_hurst(self, H: float) -> str:
        """Interpret Hurst exponent value."""
        if np.isnan(H):
            return "Cannot estimate"
        elif H < 0.3:
            return "Extremely rough - strong mean reversion"
        elif H < 0.5:
            return "Rough - mean reverting, consistent with rough volatility"
        elif H == 0.5:
            return "Brownian motion - random walk"
        elif H < 0.7:
            return "Persistent - trend continuation"
        else:
            return "Highly persistent - long memory"
    
    def plot_variogram(
        self,
        series: pd.Series,
        q: float = 2.0,
        ax=None,
        title: Optional[str] = None
    ):
        """
        Plot variogram (structure function).
        
        Args:
            series: Time series
            q: Moment order
            ax: matplotlib axes
            title: Plot title
        
        Returns:
            matplotlib axes
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = ['Arial Unicode MS']
        
        H, info = self.estimate_hurst_variogram(series, q=q)
        if np.isnan(H):
            print(f"Cannot plot: {info.get('error')}")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        log_lags = info['log_lags']
        log_moments = info['log_moments']
        slope = info['slope']
        intercept = info['intercept']
        
        ax.scatter(log_lags, log_moments, alpha=0.6, s=50, label='Data')
        fitted_line = slope * np.array(log_lags) + intercept
        ax.plot(log_lags, fitted_line, 'r--', linewidth=2, label=f'Fit (H={H:.3f})')
        
        if q == 2.0:
            ref_slope = q * 0.5
            ref_line = ref_slope * np.array(log_lags) + intercept
            ax.plot(log_lags, ref_line, 'g:', linewidth=1.5, label='H=0.5 reference', alpha=0.7)
        
        ax.set_xlabel('log(lag τ)', fontsize=12)
        ax.set_ylabel(f'log(E[|Δ|^{q}])', fontsize=12)
        if title is None:
            title = f'Hurst Estimation (Variogram)\nH = {H:.3f}, R² = {info["r_squared"]:.3f}'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return ax
    
    def plot_multiscale_comparison(
        self,
        series: pd.Series,
        q_values: Optional[List[float]] = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """Plot variograms for multiple q values (subplot mode)."""
        import matplotlib.pyplot as plt
        if q_values is None:
            q_values = [1.0, 1.5, 2.0]
        fig, axes = plt.subplots(1, len(q_values), figsize=figsize)
        if len(q_values) == 1:
            axes = [axes]
        for i, q in enumerate(q_values):
            self.plot_variogram(series, q=q, ax=axes[i], title=f'q = {q}')
        plt.tight_layout()
        return fig
    
    def plot_multiscale_overlay(
        self,
        series: pd.Series,
        q_values: Optional[List[float]] = None,
        figsize: Tuple[int, int] = (12, 8),
        show_reference: bool = True,
        title: Optional[str] = None
    ):
        """Plot multiple q variograms on one figure (overlay mode)."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        plt.rcParams['font.family'] = ['Arial Unicode MS']
        
        if q_values is None:
            q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        fig, ax = plt.subplots(figsize=figsize)
        colors = cm.rainbow(np.linspace(0, 1, len(q_values)))
        results = {}
        H_estimates = []
        
        for i, q in enumerate(q_values):
            H, info = self.estimate_hurst_variogram(series, q=q, log_space=True)
            if np.isnan(H):
                print(f"Warning: q={q} estimation failed")
                continue
            results[q] = {'H': H, 'info': info}
            H_estimates.append(H)
            log_lags = np.array(info['log_lags'])
            log_moments = np.array(info['log_moments'])
            slope, intercept = info['slope'], info['intercept']
            ax.scatter(log_lags, log_moments, color=colors[i], alpha=0.5, s=10, label=f'q={q} data')
            ax.plot(log_lags, slope * log_lags + intercept, color=colors[i], linewidth=2.5,
                    linestyle='--', label=f'q={q}: H={H:.3f}, R²={info["r_squared"]:.3f}')
        
        if show_reference and 2.0 in results:
            ref_info = results[2.0]['info']
            ref_log_lags = np.array(ref_info['log_lags'])
            ref_intercept = ref_info['intercept']
            ref_line = 1.0 * ref_log_lags + ref_intercept
            ax.plot(ref_log_lags, ref_line, 'k:', linewidth=2, alpha=0.7, label='H=0.5 ref (q=2)')
        
        ax.set_xlabel('log(lag τ)', fontsize=13)
        ax.set_ylabel('log(E[|Δ|^q])', fontsize=13)
        if title is None:
            title = f'Multi-scale Hurst Estimation\nMean H = {np.mean(H_estimates):.3f} ± {np.std(H_estimates):.3f}'
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        return fig, ax, results
