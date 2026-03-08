"""
RFSV (Rough Fractional Stochastic Volatility) predictor module.
Based on Gatheral et al. (2014) "Volatility is Rough".
Implements log-variance conditional expectation prediction via Riemann sum discretization,
with AR/HAR benchmarks.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from scipy import stats
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')


def estimate_nu_sq(log_var_series: Union[pd.Series, np.ndarray],
                   delta_max: int = 30) -> Tuple[float, dict]:
    """
    Estimate ν² (squared vol-of-vol) from log m(2, Δ) vs log Δ regression.
    
    Consistent with Hurst estimation: m(2, Δ) is defined on log σ (Gatheral Section 2.1).
    For log σ² input, internally uses log σ = 0.5 * log σ².
    Theory: log m(2, Δ) ≈ 2H·log Δ + log(ν²), so ν² = exp(intercept).
    
    Args:
        log_var_series: log σ² series (i.e. np.log(rv_uz))
        delta_max: Max lag, default 30
    
    Returns:
        (nu_sq, info_dict)
    """
    if isinstance(log_var_series, pd.Series):
        arr = (0.5 * log_var_series).values
    else:
        arr = 0.5 * np.asarray(log_var_series)
    
    if len(arr) < delta_max + 10:
        return np.nan, {'error': 'insufficient data'}
    
    log_deltas, log_m2s = [], []
    for delta in range(1, delta_max + 1):
        incr = arr[delta:] - arr[:-delta]
        if len(incr) < 5:
            continue
        m2 = np.mean(incr ** 2)
        if m2 <= 0:
            continue
        log_m2s.append(np.log(m2))
        log_deltas.append(np.log(delta))
    
    if len(log_deltas) < 3:
        return np.nan, {'error': 'insufficient valid deltas'}
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_deltas, log_m2s)
    nu_sq = np.exp(intercept)
    return nu_sq, {
        'slope': slope, 'intercept': intercept, 'r_squared': r_value ** 2,
        'nu_sq': nu_sq, 'n_deltas': len(log_deltas),
    }


class RFSVPredictor:
    """
    RFSV volatility predictor.
    
    Predicts log-variance conditional expectation via Gatheral (2014) Eq. (5.1),
    using Riemann sum discretization (no path simulation).
    """
    
    def __init__(self, H: float, nu_sq: float, delta: int = 1, window_ratio: float = 1.0):
        """
        Args:
            H: Hurst exponent
            nu_sq: ν² from m(2,Δ) regression
            delta: Prediction horizon (days)
            window_ratio: r, integration window [t - r*Δ, t]
        """
        self.H = H
        self.nu_sq = nu_sq
        self.delta = delta
        self.window_ratio = window_ratio
        self._c = gamma(1.5 - H) / (gamma(H + 0.5) * gamma(2 - 2 * H))

    def _compute_interval_weight(self, i: int) -> float:
        """
        Compute interval weight W_i for lookback step i.
        Interval u ∈ [i-0.5, i+0.5], kernel 1/((u+Δ)u^{H-1/2}).
        """
        exp_val = 1.5 - self.H
        if exp_val <= 0:
            return 0.0
        u_lo = max(0.0, i - 0.5)
        u_hi = i + 0.5
        term_hi = u_hi ** exp_val
        term_lo = u_lo ** exp_val if u_lo > 0 else 0.0
        integral_val = (term_hi - term_lo) / exp_val
        return integral_val / (i + self.delta)

    def predict_log_variance(self,
                             log_var_series: Union[pd.Series, np.ndarray],
                             t: int) -> float:
        """
        Predict log σ² at time t for horizon Δ.
        
        Based on interval integral form of Eq. (5.1):
        - Lookback i ∈ {0,1,...,n-1}, i=0 is today
        - Kernel 1/((u+Δ)u^{H-1/2}) integrated over [i-0.5, i+0.5] gives weight W_i
        - window_ratio=1, delta=1: n_max=1, uses only today
        """
        if isinstance(log_var_series, pd.Series):
            log_var_arr = log_var_series.values
        else:
            log_var_arr = np.asarray(log_var_series)
        
        n_max = max(1, int(np.ceil(self.window_ratio * self.delta)))
        n = min(t + 1, n_max)
        if n < 1:
            return np.nan
        
        weights, log_sigma_vals = [], []
        for i in range(0, n):
            idx = t - i
            if idx < 0:
                break
            w = self._compute_interval_weight(i)
            if w <= 0:
                continue
            weights.append(w)
            log_sigma_vals.append(0.5 * log_var_arr[idx])
        
        if not weights:
            return np.nan
        weights = np.array(weights)
        log_sigma_vals = np.array(log_sigma_vals)
        w_sum = np.sum(weights)
        if w_sum <= 0:
            return np.nan
        pred_log_sigma = np.dot(weights, log_sigma_vals) / w_sum
        return 2.0 * pred_log_sigma
    
    def predict_variance(self,
                        var_series: Union[pd.Series, np.ndarray],
                        t: int) -> float:
        """Predict raw variance σ²_{t+Δ}. σ̂² = exp{ log σ̂² + 2cν²Δ^{2H} }"""
        log_var = np.log(var_series)
        log_pred = self.predict_log_variance(log_var, t)
        if np.isnan(log_pred):
            return np.nan
        correction = 2 * self._c * self.nu_sq * (self.delta ** (2 * self.H))
        return np.exp(log_pred + correction)
    
    def predict_rolling(self,
                       log_var_series: Union[pd.Series, np.ndarray],
                       start_idx: int = 500) -> np.ndarray:
        """
        Rolling prediction, aligned with AR/HAR.
        
        Args:
            log_var_series: log σ² series
            start_idx: Start index (500-day warm-up per paper)
        
        Returns:
            Prediction array, length = len(series) - start_idx - delta
        """
        arr = log_var_series.values if isinstance(log_var_series, pd.Series) else np.asarray(log_var_series)
        n = len(arr)
        preds = [self.predict_log_variance(arr, t) for t in range(start_idx, n - self.delta)]
        return np.array(preds)


def predict_ar(log_var_series: Union[pd.Series, np.ndarray],
               delta: int, p: int = 5, train_window: int = 500) -> np.ndarray:
    """
    AR(p) forecast: log(σ²_{t+Δ}) = K0 + Σ Ci·log(σ²_{t-i}).
    """
    arr = log_var_series.values if isinstance(log_var_series, pd.Series) else np.asarray(log_var_series)
    n = len(arr)
    preds = []
    for k in range(train_window, n - delta):
        y_list, X_list = [], []
        for j in range(max(k - train_window, p), k - delta + 1):
            if j + delta >= n:
                continue
            y_list.append(arr[j + delta])
            X_list.append([1.0] + [arr[j - i] for i in range(p + 1)])
        if len(y_list) < p + 5:
            preds.append(np.nan)
            continue
        X, y = np.array(X_list), np.array(y_list)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            x_pred = np.array([1.0] + [arr[k - i] for i in range(p + 1)])
            preds.append(np.dot(coeffs, x_pred))
        except np.linalg.LinAlgError:
            preds.append(np.nan)
    return np.array(preds)


def predict_har(log_var_series: Union[pd.Series, np.ndarray],
               delta: int, train_window: int = 500) -> np.ndarray:
    """
    HAR(3) forecast: log(σ²_{t+Δ}) = K0 + C0·log(σ²_t) + C5·(1/5)Σlog(σ²_{t-1..t-5}) + C20·(1/20)Σlog(σ²_{t-1..t-20}).
    """
    arr = log_var_series.values if isinstance(log_var_series, pd.Series) else np.asarray(log_var_series)
    n = len(arr)
    preds = []
    for k in range(train_window, n - delta):
        y_list, X_list = [], []
        for j in range(k - train_window, k - delta + 1):
            if j + delta >= n or j < 20:
                continue
            y_list.append(arr[j + delta])
            x0, x5 = arr[j], np.mean(arr[j - 5:j]) if j >= 5 else arr[j]
            x20 = np.mean(arr[j - 20:j])
            X_list.append([1.0, x0, x5, x20])
        if len(y_list) < 10:
            preds.append(np.nan)
            continue
        X, y = np.array(X_list), np.array(y_list)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            x0 = arr[k]
            x5 = np.mean(arr[k - 5:k]) if k >= 5 else arr[k]
            x20 = np.mean(arr[k - 20:k])
            preds.append(np.dot(coeffs, [1.0, x0, x5, x20]))
        except np.linalg.LinAlgError:
            preds.append(np.nan)
    return np.array(preds)


def compute_p_ratio(actual: np.ndarray, predicted: np.ndarray, mean_log_var: float) -> float:
    """
    Compute MSE ratio P = Σ(actual - predicted)² / Σ(actual - mean)².
    P < 1 means better than naive forecast.
    """
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() < 2:
        return np.nan
    actual, predicted = actual[mask], predicted[mask]
    mse = np.mean((actual - predicted) ** 2)
    var_term = np.mean((actual - mean_log_var) ** 2)
    return mse / var_term if var_term >= 1e-12 else np.nan
