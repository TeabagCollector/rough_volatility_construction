"""
Usage example: MUZ volatility estimation and Hurst exponent computation.

Demonstrates the full analysis pipeline using the modular code.
"""

import os
import sys
sys.path.append('../models_en')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataFetcher import DataFetcher
from MUZEstimator import MUZEstimator
from HurstEstimator import HurstEstimator


# =============================================================================
# 1. Initialize modules
# =============================================================================

print("=" * 60)
print("Initializing data fetcher and estimators")
print("=" * 60)

# RiceQuant API license from env var RQDATAC_LICENSE
# Before use: export RQDATAC_LICENSE='your_license_key'
license = os.environ.get('RQDATAC_LICENSE')

data_fetcher = DataFetcher(license=license)
data_fetcher.init_connection()

muz_estimator = MUZEstimator(data_fetcher=data_fetcher)
hurst_estimator = HurstEstimator()


# =============================================================================
# 2. Compute daily variance proxy (RV^UZ)
# =============================================================================

print("\n" + "=" * 60)
print("Step 1: Compute daily variance proxy")
print("=" * 60)

symbol = '000300.XSHG'  # CSI 300 index
start_date = '20240101'
end_date = '20241231'

df_variance = muz_estimator.process_period(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    optimize_tick=True,
    adaptive_tick=False,
    verbose=True
)

print(f"\nFetched {len(df_variance)} days of variance data")
print(df_variance.head())

df_variance.to_csv('../data/rv_uz_results.csv')
print("\nVariance data saved to data/rv_uz_results.csv")


# =============================================================================
# 3. Compute Hurst exponent
# =============================================================================

print("\n" + "=" * 60)
print("Step 2: Compute Hurst exponent")
print("=" * 60)

log_variance = np.log(df_variance['rv_uz'])

# Method 1: Single q estimation
H, info = hurst_estimator.estimate_hurst_variogram(log_variance, q=2.0)
print(f"\nSingle q (q=2.0):")
print(f"  Hurst H = {H:.4f}")
print(f"  R² = {info['r_squared']:.4f}")
print(f"  Interpretation: {hurst_estimator._interpret_hurst(H)}")

# Method 2: Multi-scale robust estimation
H_mean, summary = hurst_estimator.estimate_hurst_multiscale(
    log_variance, q_values=[1.0, 1.5, 2.0]
)
print(f"\nMulti-scale:")
print(f"  Mean H = {H_mean:.4f} ± {summary['H_std']:.4f}")
for q in summary['q_values']:
    h_est = summary['results_by_q'][f'q={q}']['H']
    print(f"  q={q}: H = {h_est:.4f}")

# Method 3: Rough volatility test
roughness_test = hurst_estimator.estimate_hurst_roughness_test(
    log_variance, q=2.0, significance_level=0.05
)
print(f"\nRough volatility test:")
print(f"  H = {roughness_test['H']:.4f}")
print(f"  95% CI: [{roughness_test['ci_lower']:.4f}, {roughness_test['ci_upper']:.4f}]")
print(f"  Is rough (H < 0.5): {roughness_test['is_rough']}")
print(f"  Significant: {roughness_test['is_significant']}")
print(f"  Interpretation: {roughness_test['interpretation']}")


# =============================================================================
# 4. Visualization
# =============================================================================

print("\n" + "=" * 60)
print("Step 3: Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

ax1 = axes[0, 0]
ax1.plot(df_variance.index, df_variance['rv_uz'], linewidth=1)
ax1.set_title('Daily Realized Variance (RV^UZ)', fontsize=14)
ax1.set_xlabel('Date')
ax1.set_ylabel('RV^UZ')
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(df_variance.index, log_variance, linewidth=1, color='orange')
ax2.set_title('Log Variance log(RV^UZ)', fontsize=14)
ax2.set_xlabel('Date')
ax2.set_ylabel('log(RV^UZ)')
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
hurst_estimator.plot_variogram(log_variance, q=2.0, ax=ax3)

ax4 = axes[1, 1]
ax4.text(0.5, 0.5, f'Multi-scale results\n\nMean H = {H_mean:.4f}\nStd = {summary["H_std"]:.4f}',
         ha='center', va='center', fontsize=14, transform=ax4.transAxes)
ax4.axis('off')

plt.tight_layout()
plt.savefig('../figures/hurst_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved to figures/hurst_analysis.png")
plt.show()


# =============================================================================
# 5. Save summary
# =============================================================================

results_summary = {
    'symbol': symbol, 'start_date': start_date, 'end_date': end_date,
    'n_days': len(df_variance),
    'mean_rv_uz': df_variance['rv_uz'].mean(),
    'std_rv_uz': df_variance['rv_uz'].std(),
    'H_single_q2': H, 'H_multiscale_mean': H_mean, 'H_multiscale_std': summary['H_std'],
    'is_rough': roughness_test['is_rough'],
    'is_significant': roughness_test['is_significant'],
    'r_squared': info['r_squared']
}

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
for key, value in results_summary.items():
    print(f"{key}: {value}")
print("\nDone!")
