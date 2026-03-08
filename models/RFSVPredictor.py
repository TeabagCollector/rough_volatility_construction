"""
RFSV 波动率预测模块
基于 Gatheral et al. (2014) "Volatility is Rough" 的预测框架
实现对数方差条件期望的黎曼和离散化预测，以及 AR/HAR 基准对比
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
    从 log m(2, Δ) 对 log Δ 的回归中估计 ν² (vol-of-vol 的平方)
    
    与 Hurst 估计一致，m(2, Δ) 定义在 log σ 上（Gatheral Section 2.1）。
    输入 log σ² 时内部转为 log σ = 0.5 * log σ²。
    理论关系：log m(2, Δ) ≈ 2H·log Δ + log(ν²)，截距的指数即为 ν²。
    
    Args:
        log_var_series: log σ² 序列（即 np.log(rv_uz)）
        delta_max: 最大滞后期，默认 30
    
    Returns:
        (nu_sq, info_dict): ν² 和回归信息
    """
    # m(2, Δ) 定义在 log σ 上（与 Hurst 估计一致），log σ = 0.5 * log σ²
    if isinstance(log_var_series, pd.Series):
        arr = (0.5 * log_var_series).values
    else:
        arr = 0.5 * np.asarray(log_var_series)
    
    if len(arr) < delta_max + 10:
        return np.nan, {'error': 'insufficient data'}
    
    log_deltas = []
    log_m2s = []
    
    for delta in range(1, delta_max + 1):
        # delta 步长增量：arr[t+Δ]-arr[t]，非 np.diff(., n=Δ) 的 Δ 阶差分
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
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_deltas, log_m2s
    )
    nu_sq = np.exp(intercept)
    
    info = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'nu_sq': nu_sq,
        'n_deltas': len(log_deltas),
    }
    return nu_sq, info


class RFSVPredictor:
    """
    RFSV 波动率预测器
    
    基于 Gatheral (2014) 公式 (5.1) 的对数方差条件期望预测
    使用黎曼和离散化，无需路径模拟
    """
    
    def __init__(self,
                 H: float,
                 nu_sq: float,
                 delta: int = 1,
                 window_ratio: float = 1.0):
        """
        Args:
            H: Hurst 指数
            nu_sq: ν²，从 m(2,Δ) 回归截距得到
            delta: 预测 horizon（天数）
            window_ratio: r，积分窗口 [t - r*Δ, t]
        """
        self.H = H
        self.nu_sq = nu_sq
        self.delta = delta
        self.window_ratio = window_ratio
        
        # 修正系数 c = Γ(3/2-H) / (Γ(H+1/2) * Γ(2-2H))，用于方差预测
        self._c = gamma(1.5 - H) / (gamma(H + 0.5) * gamma(2 - 2 * H))

    def _compute_interval_weight(self, i: int) -> float:
        """
        计算回溯步长 i 对应的区间积分权重 W_i。

        区间 u ∈ [i-0.5, i+0.5]，核函数 1/((u+Δ)u^{H-1/2})。
        近似：W_i ≈ ((i+0.5)^{1.5-H} - u_lo^{1.5-H}) / ((i+Δ)(1.5-H))
        - i=0（今天）：u ∈ [0, 0.5]，u_lo=0，积分收敛
        - i≥1：u_lo = max(0, i-0.5)，避免原点奇异性
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
        在时刻 t 预测未来 Δ 日的 log σ²

        基于「观测值势力范围」的区间积分形式（公式 5.1 离散化）：
        - 回溯步长 i ∈ {0, 1, ..., n-1}，i=0 为今天，对应 u ∈ [i-0.5, i+0.5]
        - 核函数 1/((u+Δ)u^{H-1/2}) 在区间内积分得权重 W_i
        - window_ratio=1 且 delta=1 时，n_max=1，仅用今天（1 日）数据
        - 在 logσ 空间加权（输入 0.5*logσ²），归一化后还原为 logσ²
        """
        if isinstance(log_var_series, pd.Series):
            log_var_arr = log_var_series.values
        else:
            log_var_arr = np.asarray(log_var_series)
        
        n_max = max(1, int(np.ceil(self.window_ratio * self.delta)))
        n = min(t + 1, n_max)
        
        if n < 1:
            return np.nan
        
        weights = []
        log_sigma_vals = []
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
        """
        预测原始方差 σ²_{t+Δ}
        
        σ̂² = exp{ log σ̂² + 2cν²Δ^{2H} }
        """
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
        滚动预测，返回与 AR/HAR 对齐的预测序列
        
        Args:
            log_var_series: log σ² 序列
            start_idx: 起始索引（与论文 500 日 warm-up 一致）
        
        Returns:
            预测值数组，长度 = len(series) - start_idx - delta
        """
        if isinstance(log_var_series, pd.Series):
            arr = log_var_series.values
        else:
            arr = np.asarray(log_var_series)
        
        n = len(arr)
        preds = []
        for t in range(start_idx, n - self.delta):
            p = self.predict_log_variance(arr, t)
            preds.append(p)
        
        return np.array(preds)


def predict_ar(log_var_series: Union[pd.Series, np.ndarray],
               delta: int,
               p: int = 5,
               train_window: int = 500) -> np.ndarray:
    """
    AR(p) 预测：log(σ²_{t+Δ}) = K0 + Σ Ci·log(σ²_{t-i})
    
    Args:
        log_var_series: log σ² 序列
        delta: 预测 horizon
        p: AR 阶数
        train_window: 滚动估计窗口
    
    Returns:
        预测序列
    """
    if isinstance(log_var_series, pd.Series):
        arr = log_var_series.values
    else:
        arr = np.asarray(log_var_series)
    
    n = len(arr)
    preds = []
    
    for k in range(train_window, n - delta):
        # 构建回归：y = log σ²_{k+Δ}, X = [1, log σ²_{k}, log σ²_{k-1}, ..., log σ²_{k-p}]
        y_list = []
        X_list = []
        for j in range(max(k - train_window, p), k - delta + 1):
            if j + delta >= n:
                continue
            y = arr[j + delta]
            x = [1.0] + [arr[j - i] for i in range(p + 1)]
            y_list.append(y)
            X_list.append(x)
        
        if len(y_list) < p + 5:
            preds.append(np.nan)
            continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            x_pred = np.array([1.0] + [arr[k - i] for i in range(p + 1)])
            preds.append(np.dot(coeffs, x_pred))
        except np.linalg.LinAlgError:
            preds.append(np.nan)
    
    return np.array(preds)


def predict_har(log_var_series: Union[pd.Series, np.ndarray],
               delta: int,
               train_window: int = 500) -> np.ndarray:
    """
    HAR(3) 预测：log(σ²_{t+Δ}) = K0 + C0·log(σ²_t) + C5·(1/5)Σlog(σ²_{t-1..t-5}) + C20·(1/20)Σlog(σ²_{t-1..t-20})
    
    Args:
        log_var_series: log σ² 序列
        delta: 预测 horizon
        train_window: 滚动估计窗口
    
    Returns:
        预测序列
    """
    if isinstance(log_var_series, pd.Series):
        arr = log_var_series.values
    else:
        arr = np.asarray(log_var_series)
    
    n = len(arr)
    preds = []
    
    for k in range(train_window, n - delta):
        y_list = []
        X_list = []
        for j in range(k - train_window, k - delta + 1):
            if j + delta >= n or j < 20:
                continue
            y = arr[j + delta]
            x0 = arr[j]
            x5 = np.mean(arr[j - 5:j]) if j >= 5 else arr[j]
            x20 = np.mean(arr[j - 20:j])
            X_list.append([1.0, x0, x5, x20])
            y_list.append(y)
        
        if len(y_list) < 10:
            preds.append(np.nan)
            continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            x0 = arr[k]
            x5 = np.mean(arr[k - 5:k]) if k >= 5 else arr[k]
            x20 = np.mean(arr[k - 20:k])
            x_pred = np.array([1.0, x0, x5, x20])
            preds.append(np.dot(coeffs, x_pred))
        except np.linalg.LinAlgError:
            preds.append(np.nan)
    
    return np.array(preds)


def compute_p_ratio(actual: np.ndarray,
                   predicted: np.ndarray,
                   mean_log_var: float) -> float:
    """
    计算 MSE 比例 P
    
    P = Σ(actual - predicted)² / Σ(actual - mean)²
    P < 1 表示优于 naive 预测
    """
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() < 2:
        return np.nan
    actual = actual[mask]
    predicted = predicted[mask]
    mse = np.mean((actual - predicted) ** 2)
    var_term = np.mean((actual - mean_log_var) ** 2)
    if var_term < 1e-12:
        return np.nan
    return mse / var_term
