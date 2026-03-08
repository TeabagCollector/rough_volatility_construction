# Rough Volatility & IV Construction

Implementation of rough volatility modeling and implied variance construction, based on Gatheral et al. (2014) "Volatility is Rough" and Robert & Rosenbaum (2011) MUZ model.

## Features

- **MUZ Estimator**: Model with Uncertainty Zones (Robert & Rosenbaum 2011) — extracts instantaneous variance proxy (RV^UZ) from high-frequency tick data
- **Hurst Estimator**: Variogram-based Hurst exponent estimation for fractional Brownian motion; multi-scale robust estimation; roughness test (H < 0.5)
- **RFSV Predictor**: Rough Fractional Stochastic Volatility forecast via Riemann sum discretization (Gatheral Eq. 5.1); AR/HAR benchmarks; P-ratio evaluation

## Project Structure

```
├── models/           # Chinese version
├── models_en/        # English version
├── Notebooks/        # Chinese notebooks
├── Notebooks_en/     # English notebooks
├── examples/         # Chinese usage example
├── examples_en/      # English usage example
├── data/             # Variance proxy & Oxford-Man data
├── requirements.txt
├── .env.example      # Env var template
└── README.md
```

## Setup

```bash
# Clone and install
pip install -r requirements.txt

# Set RiceQuant API license (required for tick data)
export RQDATAC_LICENSE='your_license_key'
# Or copy .env.example to .env and fill in your key
```

## Usage

**English version** (recommended for reproducibility):

```python
import os
import sys
sys.path.append('models_en')

from DataFetcher import DataFetcher
from MUZEstimator import MUZEstimator
from HurstEstimator import HurstEstimator

# License from env
license = os.environ.get('RQDATAC_LICENSE')
data_fetcher = DataFetcher(license=license)
data_fetcher.init_connection()

muz = MUZEstimator(data_fetcher=data_fetcher)
df_variance = muz.process_period(
    symbol='000300.XSHG',
    start_date='20240101',
    end_date='20241231',
    optimize_tick=True,
    verbose=True
)

hurst = HurstEstimator()
H, info = hurst.estimate_hurst_variogram(np.log(df_variance['rv_uz']), q=2.0)
print(f"Hurst H = {H:.4f}")
```

Or run the notebooks in `Notebooks_en/` for the full pipeline.

## Data

- **CSI 300 tick data**: Requires RiceQuant (rqdatac) license; fetched via `DataFetcher`
- **Oxford-Man Realized Volatility**: Public dataset in `data/oxfordmanrealizedvolatilityindices.csv` (AEX, etc.)
- **Precomputed variance proxy**: `data/variance_proxy_*.csv` (optional; can be regenerated)

## References

- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2014). *Volatility is rough.* Quantitative Finance.
- Robert, C. Y., & Rosenbaum, M. (2011). *A new approach for the dynamics of ultra-high-frequency data: The model with uncertainty zones.* Journal of Financial Econometrics.

## License

MIT (or specify your license)
