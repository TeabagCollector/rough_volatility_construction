"""
使用示例：MUZ波动率估计和Hurst指数计算

展示如何使用模块化后的代码进行完整的分析流程
"""

import os
import sys
sys.path.append('../models')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataFetcher import DataFetcher
from MUZEstimator import MUZEstimator
from HurstEstimator import HurstEstimator


# =============================================================================
# 1. 初始化模块
# =============================================================================

print("=" * 60)
print("初始化数据获取器和估计器")
print("=" * 60)

# 米筐 API license 从环境变量 RQDATAC_LICENSE 读取
# 使用前请设置: export RQDATAC_LICENSE='your_license_key'
license = os.environ.get('RQDATAC_LICENSE')

# 初始化（若 license 为 None，init_connection 会尝试从环境变量读取）
data_fetcher = DataFetcher(license=license)
data_fetcher.init_connection()

muz_estimator = MUZEstimator(data_fetcher=data_fetcher)
hurst_estimator = HurstEstimator()


# =============================================================================
# 2. 计算日度方差代理值 (RV^UZ)
# =============================================================================

print("\n" + "=" * 60)
print("步骤1：计算日度方差代理值")
print("=" * 60)

# 设置参数
symbol = '000300.XSHG'  # 沪深300指数
start_date = '20240101'
end_date = '20241231'

# 批量处理
df_variance = muz_estimator.process_period(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    optimize_tick=True,
    adaptive_tick=False,
    verbose=True
)

print(f"\n成功获取 {len(df_variance)} 天的方差数据")
print(df_variance.head())

# 保存结果
df_variance.to_csv('../data/rv_uz_results.csv')
print("\n✓ 方差数据已保存到 data/rv_uz_results.csv")


# =============================================================================
# 3. 计算 Hurst 指数
# =============================================================================

print("\n" + "=" * 60)
print("步骤2：计算 Hurst 指数")
print("=" * 60)

# 使用对数方差序列
log_variance = np.log(df_variance['rv_uz'])

# 方法1：单一q值估计
H, info = hurst_estimator.estimate_hurst_variogram(
    log_variance, 
    q=2.0
)

print(f"\n单一q值估计 (q=2.0):")
print(f"  Hurst指数 H = {H:.4f}")
print(f"  R² = {info['r_squared']:.4f}")
print(f"  解释: {hurst_estimator._interpret_hurst(H)}")

# 方法2：多尺度鲁棒估计
H_mean, summary = hurst_estimator.estimate_hurst_multiscale(
    log_variance,
    q_values=[1.0, 1.5, 2.0]
)

print(f"\n多尺度估计:")
print(f"  平均 Hurst指数 = {H_mean:.4f} ± {summary['H_std']:.4f}")
for q in summary['q_values']:
    h_est = summary['results_by_q'][f'q={q}']['H']
    print(f"  q={q}: H = {h_est:.4f}")

# 方法3：Rough Volatility 特性检验
roughness_test = hurst_estimator.estimate_hurst_roughness_test(
    log_variance,
    q=2.0,
    significance_level=0.05
)

print(f"\nRough Volatility 检验:")
print(f"  H = {roughness_test['H']:.4f}")
print(f"  95% 置信区间: [{roughness_test['ci_lower']:.4f}, {roughness_test['ci_upper']:.4f}]")
print(f"  是否粗糙 (H < 0.5): {roughness_test['is_rough']}")
print(f"  显著性: {roughness_test['is_significant']}")
print(f"  解释: {roughness_test['interpretation']}")


# =============================================================================
# 4. 可视化
# =============================================================================

print("\n" + "=" * 60)
print("步骤3：可视化结果")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 子图1：日度方差序列
ax1 = axes[0, 0]
ax1.plot(df_variance.index, df_variance['rv_uz'], linewidth=1)
ax1.set_title('日度实现方差 (RV^UZ)', fontsize=14)
ax1.set_xlabel('日期')
ax1.set_ylabel('RV^UZ')
ax1.grid(True, alpha=0.3)

# 子图2：对数方差序列
ax2 = axes[0, 1]
ax2.plot(df_variance.index, log_variance, linewidth=1, color='orange')
ax2.set_title('对数方差序列 log(RV^UZ)', fontsize=14)
ax2.set_xlabel('日期')
ax2.set_ylabel('log(RV^UZ)')
ax2.grid(True, alpha=0.3)

# 子图3：变差图 (q=2)
ax3 = axes[1, 0]
hurst_estimator.plot_variogram(log_variance, q=2.0, ax=ax3)

# 子图4：多尺度对比
ax4 = axes[1, 1]
# 这里可以添加不同q值的对比图
ax4.text(0.5, 0.5, f'多尺度估计结果\n\n平均 H = {H_mean:.4f}\n标准差 = {summary["H_std"]:.4f}',
         ha='center', va='center', fontsize=14, transform=ax4.transAxes)
ax4.axis('off')

plt.tight_layout()
plt.savefig('../figures/hurst_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存到 figures/hurst_analysis.png")

plt.show()


# =============================================================================
# 5. 保存完整结果
# =============================================================================

results_summary = {
    'symbol': symbol,
    'start_date': start_date,
    'end_date': end_date,
    'n_days': len(df_variance),
    'mean_rv_uz': df_variance['rv_uz'].mean(),
    'std_rv_uz': df_variance['rv_uz'].std(),
    'H_single_q2': H,
    'H_multiscale_mean': H_mean,
    'H_multiscale_std': summary['H_std'],
    'is_rough': roughness_test['is_rough'],
    'is_significant': roughness_test['is_significant'],
    'r_squared': info['r_squared']
}

# 打印总结
print("\n" + "=" * 60)
print("分析总结")
print("=" * 60)
for key, value in results_summary.items():
    print(f"{key}: {value}")

print("\n✓ 分析完成!")
