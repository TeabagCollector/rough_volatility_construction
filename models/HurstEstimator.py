"""
Hurst指数估计模块
基于Gatheral et al. (2014) "Volatility is Rough"的方法实现
使用变差分析和q阶矩回归估计分数布朗运动的Hurst指数
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class HurstEstimator:
    """
    Hurst 指数估计器
    
    基于 Gatheral et al. (2014) "Volatility is Rough" 的方法
    使用变差分析和 q 阶矩回归估计分数布朗运动的 Hurst 指数
    
    主要方法：
    - estimate_hurst_variogram: 单一q值的变差法估计
    - estimate_hurst_multiscale: 多q值的鲁棒估计
    - estimate_hurst_roughness_test: Rough Volatility特性检验
    - plot_variogram: 可视化变差图
    """
    
    def __init__(self):
        """初始化 Hurst 估计器"""
        pass
    
    def _calculate_qth_moment(
        self,
        series: pd.Series,
        lag: int,
        q: float = 2.0
    ) -> float:
        """
        计算时间序列的 q 阶增量矩
        
        E[|X(t+τ) - X(t)|^q]
        
        Args:
            series: 时间序列
            lag: 滞后期 τ
            q: 矩的阶数（默认2表示方差）
        
        Returns:
            q 阶矩的估计值
        """
        if len(series) <= lag:
            return np.nan
        
        increments = series.diff(lag).dropna()
        
        if len(increments) == 0:
            return np.nan
        
        qth_moment = np.mean(np.abs(increments) ** q)
        
        return qth_moment
    
    def estimate_hurst_variogram(
        self,
        series: pd.Series,
        lags: Optional[List[int]] = None,
        q: float = 2.0,
        log_space: bool = True
    ) -> Tuple[float, Dict]:
        """
        使用变差法（Variogram/Structure Function）估计 Hurst 指数
        
        基于理论关系：E[|X(t+τ) - X(t)|^q] ~ τ^(qH)
        对数化后：log(E[|ΔX_τ|^q]) = qH * log(τ) + const
        
        通过线性回归估计斜率 qH，得到 H = slope / q
        
        Args:
            series: 时间序列（通常是对数方差的时间序列）
            lags: 滞后期列表，如果不提供则自动生成
            q: 矩的阶数，q=2 对应标准变差
            log_space: 是否在对数空间均匀取样滞后期
        
        Returns:
            (H, info_dict): Hurst指数和详细信息
        """
        if len(series) < 10:
            return np.nan, {'error': 'insufficient data'}
        
        # 自动生成滞后期
        if lags is None:
            max_lag = min(len(series) // 3, 250)  # 最多取1/3长度或250天
            if log_space:
                # 对数空间均匀取样
                lags = np.unique(np.logspace(0, np.log10(max_lag), 50).astype(int))
            else:
                # 线性空间均匀取样
                lags = np.arange(1, max_lag, max(1, max_lag // 50))
        
        # 计算各个滞后期的 q 阶矩
        moments = []
        valid_lags = []
        
        for lag in lags:
            moment = self._calculate_qth_moment(series, lag, q)
            if not np.isnan(moment) and moment > 0:
                moments.append(moment)
                valid_lags.append(lag)
        
        if len(valid_lags) < 3:
            return np.nan, {'error': 'insufficient valid lags'}
        
        # 对数-对数回归
        log_lags = np.log(valid_lags)
        log_moments = np.log(moments)
        
        # 线性回归：log(moment) = slope * log(lag) + intercept
        # slope = qH, 所以 H = slope / q
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_moments)
        
        H = slope / q
        
        info = {
            'H': H,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'q': q,
            'n_lags': len(valid_lags),
            'lags': valid_lags,
            'moments': moments,
            'log_lags': log_lags.tolist(),
            'log_moments': log_moments.tolist()
        }
        
        return H, info
    
    def estimate_hurst_multiscale(
        self,
        series: pd.Series,
        q_values: Optional[List[float]] = None,
        lags: Optional[List[int]] = None
    ) -> Tuple[float, Dict]:
        """
        多尺度 Hurst 指数估计
        
        使用多个 q 值进行估计，取平均值提高鲁棒性
        
        Args:
            series: 时间序列
            q_values: q 值列表，默认 [1.0, 1.5, 2.0]
            lags: 滞后期列表
        
        Returns:
            (H_mean, info_dict): 平均 Hurst 指数和详细信息
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
        
        H_mean = np.mean(H_estimates)
        H_std = np.std(H_estimates)
        
        summary = {
            'H_mean': H_mean,
            'H_std': H_std,
            'H_estimates': H_estimates,
            'q_values': q_values,
            'n_valid': len(H_estimates),
            'results_by_q': results
        }
        
        return H_mean, summary
    
    def estimate_hurst_roughness_test(
        self,
        variance_series: pd.Series,
        q: float = 2.0,
        significance_level: float = 0.05
    ) -> Dict:
        """
        Rough Volatility 特性检验
        
        检验 H 是否显著小于 0.5（rough 的标志）
        
        Args:
            variance_series: 方差时间序列（通常是 log(RV) 或 RV）
            q: 矩的阶数
            significance_level: 显著性水平
        
        Returns:
            包含检验结果的字典
        """
        H, info = self.estimate_hurst_variogram(variance_series, q=q)
        
        if np.isnan(H):
            return {'error': 'failed to estimate H', 'info': info}
        
        # 检验 H < 0.5
        is_rough = H < 0.5
        
        # 计算 H 的置信区间（简化估计，基于回归标准误）
        std_err = info.get('std_err', np.nan)
        if not np.isnan(std_err):
            # H = slope / q, 所以 se(H) = se(slope) / q
            H_std_err = std_err / q
            
            # 95% 置信区间
            t_critical = stats.t.ppf(1 - significance_level/2, info['n_lags'] - 2)
            ci_lower = H - t_critical * H_std_err
            ci_upper = H + t_critical * H_std_err
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        
        result = {
            'H': H,
            'is_rough': is_rough,
            'is_significant': ci_upper < 0.5,  # 置信区间上界也小于0.5
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r_squared': info.get('r_squared'),
            'p_value': info.get('p_value'),
            'n_lags': info.get('n_lags'),
            'interpretation': self._interpret_hurst(H)
        }
        
        return result
    
    def _interpret_hurst(self, H: float) -> str:
        """
        解释 Hurst 指数的含义
        
        Args:
            H: Hurst 指数
        
        Returns:
            解释文本
        """
        if np.isnan(H):
            return "无法估计"
        elif H < 0.3:
            return "极度粗糙 (Extremely Rough) - 强均值回复"
        elif H < 0.5:
            return "粗糙 (Rough) - 均值回复，符合 Rough Volatility 理论"
        elif H == 0.5:
            return "标准布朗运动 (Brownian Motion) - 随机游走"
        elif H < 0.7:
            return "持续性 (Persistent) - 趋势延续"
        else:
            return "强持续性 (Highly Persistent) - 长记忆"
    
    def plot_variogram(
        self,
        series: pd.Series,
        q: float = 2.0,
        ax=None,
        title: Optional[str] = None
    ):
        """
        绘制变差图（Structure Function）
        
        Args:
            series: 时间序列
            q: 矩的阶数
            ax: matplotlib axes 对象
            title: 图表标题
        
        Returns:
            matplotlib axes 对象
        """
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = ['Arial Unicode MS']
        
        H, info = self.estimate_hurst_variogram(series, q=q)
        
        if np.isnan(H):
            print(f"无法绘制: {info.get('error')}")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        log_lags = info['log_lags']
        log_moments = info['log_moments']
        slope = info['slope']
        intercept = info['intercept']
        
        # 散点图：实际数据
        ax.scatter(log_lags, log_moments, alpha=0.6, s=50, label='数据点')
        
        # 拟合线
        fitted_line = slope * np.array(log_lags) + intercept
        ax.plot(log_lags, fitted_line, 'r--', linewidth=2, 
                label=f'拟合线 (H={H:.3f})')
        
        # 参考线：H=0.5
        if q == 2.0:
            ref_slope = q * 0.5
            ref_line = ref_slope * np.array(log_lags) + intercept
            ax.plot(log_lags, ref_line, 'g:', linewidth=1.5, 
                    label='H=0.5 参考线', alpha=0.7)
        
        ax.set_xlabel('log(滞后期 τ)', fontsize=12)
        ax.set_ylabel(f'log(E[|Δ|^{q}])', fontsize=12)
        
        if title is None:
            title = f'Hurst 指数估计 (Variogram)\nH = {H:.3f}, R² = {info["r_squared"]:.3f}'
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
        """
        多尺度对比可视化 - 分图模式
        
        绘制不同q值下的变差图对比（每个q一个子图）
        
        Args:
            series: 时间序列
            q_values: q值列表
            figsize: 图表尺寸
        
        Returns:
            matplotlib figure 对象
        """
        import matplotlib.pyplot as plt
        
        if q_values is None:
            q_values = [1.0, 1.5, 2.0]
        
        fig, axes = plt.subplots(1, len(q_values), figsize=figsize)
        
        if len(q_values) == 1:
            axes = [axes]
        
        for i, q in enumerate(q_values):
            self.plot_variogram(series, q=q, ax=axes[i], 
                              title=f'q = {q}')
        
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
        """
        多尺度叠加可视化 - 单图模式
        
        在同一张图上绘制多个q值的变差图和回归线
        
        Args:
            series: 时间序列
            q_values: q值列表，默认 [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            figsize: 图表尺寸
            show_reference: 是否显示H=0.5参考线
            title: 图表标题
        
        Returns:
            (fig, ax, results): matplotlib对象和所有q值的估计结果
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        plt.rcParams['font.family'] = ['Arial Unicode MS']
        
        if q_values is None:
            q_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用不同颜色
        colors = cm.rainbow(np.linspace(0, 1, len(q_values)))
        
        results = {}
        H_estimates = []
        
        # 对每个q值计算并绘制
        for i, q in enumerate(q_values):
            H, info = self.estimate_hurst_variogram(series, q=q, log_space=True)
            
            if np.isnan(H):
                print(f"警告: q={q} 估计失败")
                continue
            
            results[q] = {'H': H, 'info': info}
            H_estimates.append(H)
            
            log_lags = np.array(info['log_lags'])
            log_moments = np.array(info['log_moments'])
            slope = info['slope']
            intercept = info['intercept']
            r_squared = info['r_squared']
            
            # 散点图
            ax.scatter(log_lags, log_moments, 
                      color=colors[i], alpha=0.5, s=10,
                      label=f'q={q} 数据')
            
            # 拟合线
            fitted_line = slope * log_lags + intercept
            ax.plot(log_lags, fitted_line, 
                   color=colors[i], linewidth=2.5, linestyle='--',
                   label=f'q={q}: H={H:.3f}, R²={r_squared:.3f}')
        
        # 添加H=0.5参考线（使用q=2.0作为参考）
        if show_reference and 2.0 in results:
            ref_info = results[2.0]['info']
            ref_log_lags = np.array(ref_info['log_lags'])
            ref_intercept = ref_info['intercept']
            ref_slope_05 = 2.0 * 0.5  # q*H = 2*0.5 = 1.0
            ref_line = ref_slope_05 * ref_log_lags + ref_intercept
            ax.plot(ref_log_lags, ref_line, 
                   'k:', linewidth=2, alpha=0.7,
                   label='H=0.5 参考线 (q=2)')
        
        # 设置标签和标题
        ax.set_xlabel('log(滞后期 τ)', fontsize=13)
        ax.set_ylabel('log(E[|Δ|^q])', fontsize=13)
        
        if title is None:
            H_mean = np.mean(H_estimates)
            H_std = np.std(H_estimates)
            title = f'多尺度 Hurst 指数估计\n平均 H = {H_mean:.3f} ± {H_std:.3f}'
        ax.set_title(title, fontsize=15, fontweight='bold')
        
        # 图例（放在右侧）
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        return fig, ax, results
