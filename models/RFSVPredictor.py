"""
RFSV 波动率预测模块
基于 Gatheral et al. (2014) "Volatility is Rough" 的预测框架
实现对数方差条件期望的黎曼和离散化预测，以及 AR/HAR 基准对比
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from scipy.integrate import quad
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

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
                 window_ratio: float = 1.0,
                 weight_mode: str = "hybrid",
                 exact_lag_cutoff: int = 8,
                 use_second_order_tail: bool = False,
                 second_order_start: int = 12):
        """
        Args:
            H: Hurst 指数
            nu_sq: ν²，从 m(2,Δ) 回归截距得到
            delta: 预测 horizon（天数）
            window_ratio: r，积分窗口 [t - r*Δ, t]
            weight_mode: 权重计算模式，可选 "approx"/"hybrid"/"exact"
            exact_lag_cutoff: hybrid 模式下用精确积分的最大滞后 i
            use_second_order_tail: 是否在大滞后区间启用二阶尾部修正
            second_order_start: 二阶尾部修正启用的最小滞后 i
        """
        self.H = H
        self.nu_sq = nu_sq
        self.delta = delta
        self.window_ratio = window_ratio
        self.weight_mode = str(weight_mode).lower().strip()
        self.exact_lag_cutoff = int(exact_lag_cutoff)
        self.use_second_order_tail = bool(use_second_order_tail)
        self.second_order_start = int(second_order_start)
        
        # 修正系数 c = Γ(3/2-H) / (Γ(H+1/2) * Γ(2-2H))，用于方差预测
        self._c = gamma(1.5 - H) / (gamma(H + 0.5) * gamma(2 - 2 * H))
        self._weight_cache: Dict[Tuple[float, int, float, int, str, int, bool, int], np.ndarray] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        if not (0.0 < self.H < 0.5):
            raise ValueError("H 必须在 (0, 0.5) 区间内。")
        if self.delta <= 0:
            raise ValueError("delta 必须为正整数。")
        if self.window_ratio <= 0:
            raise ValueError("window_ratio 必须为正数。")
        if self.weight_mode not in {"approx", "hybrid", "exact"}:
            raise ValueError("weight_mode 必须是 'approx'、'hybrid' 或 'exact'。")
        if self.exact_lag_cutoff < 0:
            raise ValueError("exact_lag_cutoff 不能为负数。")
        if self.second_order_start < 0:
            raise ValueError("second_order_start 不能为负数。")

    def _compute_interval_weight_approx(self, i: int) -> float:
        """
        计算回溯步长 i 对应的区间积分权重 W_i。

        区间 u ∈ [i-0.5, i+0.5]，核函数 1/((u+Δ)u^{H+1/2})。
        近似：W_i ≈ ((i+0.5)^{0.5-H} - u_lo^{0.5-H}) / ((i+Δ)(0.5-H))
        - i=0（今天）：u ∈ [0, 0.5]，u_lo=0，积分收敛
        - i≥1：u_lo = max(0, i-0.5)，避免原点奇异性
        """
        exp_val = 0.5 - self.H
        if exp_val <= 0:
            return 0.0
        u_lo = max(0.0, i - 0.5)
        u_hi = i + 0.5
        term_hi = u_hi ** exp_val
        term_lo = u_lo ** exp_val if u_lo > 0 else 0.0
        integral_val = (term_hi - term_lo) / exp_val
        return integral_val / (i + self.delta)

    def _compute_interval_weight_exact(self, i: int) -> float:
        """
        数值积分计算精确区间权重：
        W_i = ∫_{u_lo}^{u_hi} 1 / ((u+Δ)u^{H+1/2}) du
        """
        u_lo = max(0.0, i - 0.5)
        u_hi = i + 0.5

        def integrand(u: float) -> float:
            return 1.0 / ((u + self.delta) * (u ** (self.H + 0.5)))

        points = [0.0] if i == 0 else None
        val, _ = quad(integrand, u_lo, u_hi, points=points, limit=200)
        return float(val)

    def _compute_second_order_tail_correction(self, i: int) -> float:
        """
        大滞后尾部二阶修正（针对 1/(u+Δ) 的二阶展开近似）。
        该修正在 i 足够大时有效。
        """
        if i <= 0:
            return 0.0
        a = self.H + 0.5
        # 对称区间 [i-0.5, i+0.5] 上近似：∫(u-i)^2 u^{-a} du ≈ i^{-a} / 12
        return (i ** (-a)) / (12.0 * ((i + self.delta) ** 3))

    def _compute_interval_weight(self, i: int) -> float:
        if self.weight_mode == "exact":
            return self._compute_interval_weight_exact(i)

        if self.weight_mode == "approx":
            w = self._compute_interval_weight_approx(i)
            if self.use_second_order_tail and i >= self.second_order_start:
                w += self._compute_second_order_tail_correction(i)
            return w

        # hybrid: 小滞后精确，大滞后快速近似
        if i <= self.exact_lag_cutoff:
            return self._compute_interval_weight_exact(i)

        w = self._compute_interval_weight_approx(i)
        if self.use_second_order_tail and i >= self.second_order_start:
            w += self._compute_second_order_tail_correction(i)
        return w

    def _get_weight_vector(self, n: int) -> np.ndarray:
        """
        获取长度为 n 的权重向量（含缓存），用于加速滚动预测。
        """
        key = (
            round(float(self.H), 12),
            int(self.delta),
            round(float(self.window_ratio), 8),
            int(n),
            self.weight_mode,
            int(self.exact_lag_cutoff),
            bool(self.use_second_order_tail),
            int(self.second_order_start),
        )
        if key not in self._weight_cache:
            w = np.array([self._compute_interval_weight(i) for i in range(n)], dtype=float)
            self._weight_cache[key] = w
        return self._weight_cache[key]

    def predict_log_variance(self,
                             log_var_series: Union[pd.Series, np.ndarray],
                             t: int) -> float:
        """
        在时刻 t 预测未来 Δ 日的 log σ²

        基于「观测值势力范围」的区间积分形式（公式 5.1 离散化）：
        - 回溯步长 i ∈ {0, 1, ..., n-1}，i=0 为今天，对应 u ∈ [i-0.5, i+0.5]
        - 核函数 1/((u+Δ)u^{H+1/2}) 在区间内积分得权重 W_i
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
        
        # 先取缓存权重，再对有效正权重做筛选
        all_weights = self._get_weight_vector(n)
        valid_mask = all_weights > 0
        if not np.any(valid_mask):
            return np.nan

        weights = all_weights[valid_mask]
        i_idx = np.arange(n)[valid_mask]
        log_sigma_vals = 0.5 * log_var_arr[t - i_idx]
        w_sum = float(np.sum(weights))
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
