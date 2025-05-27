# Market Regime Detection Library

A comprehensive library for detecting market regimes using multiple methods, including K-means Clustering, Hidden Markov Models, Bayesian Changepoint Detection, Ruptures-based Changepoint Detection, and Ensemble Methods.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import pandas as pd
from src.regime_detection import get_detector, RegimeDetectorType

# Load your data (OHLCV format)
df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)

# Create a detector
detector = get_detector(RegimeDetectorType.KMEANS)

# Fit the model
regime_df = detector.fit(df)

# Plot the regimes
dashboard_path = detector.create_regime_dashboard(regime_df)
print(f"Dashboard created at: {dashboard_path}")

# Predict on new data
new_data = df.iloc[-100:]  # As an example
prediction = detector.predict(new_data)
```

## Available Detectors

### K-means Clustering

Uses unsupervised clustering to identify distinct market regimes based on technical indicators.

```python
from src.regime_detection import KMeansRegimeDetector

# Parameters
detector = KMeansRegimeDetector(
    n_regimes=4,              # Number of regimes (or None for automatic)
    lookback_period=20,       # Lookback period for feature calculation
    max_regimes=5             # Maximum number of regimes if automatic
)
```

### Hidden Markov Models (HMM)

Utilizes Hidden Markov Models to identify market states based on the transition probabilities between regimes.

```python
from src.regime_detection import HMMRegimeDetector

# Parameters
detector = HMMRegimeDetector(
    n_regimes=4,              # Number of regimes
    lookback_period=20,       # Lookback period for feature calculation
    n_iter=1000               # Number of iterations for HMM training
)
```

### Bayesian Changepoint Detection

Uses PyMC for Bayesian inference to detect structural changes in the market with uncertainty estimates.

```python
from src.regime_detection import BayesianRegimeDetector

# Parameters
detector = BayesianRegimeDetector(
    n_regimes=None,           # Number of regimes (or None for automatic)
    mcmc_samples=1000,        # Number of samples for MCMC
    feature='log_returns'     # Feature to use for changepoint detection
)
```

### Ruptures Changepoint Detection

Uses the ruptures library for efficient changepoint detection to identify shifts in market dynamics.

```python
from src.regime_detection import RupturesRegimeDetector

# Parameters
detector = RupturesRegimeDetector(
    penalty=10,               # Penalty value (higher = fewer changepoints)
    min_size=20,              # Minimum segment size
    method='dynp'             # Method: 'dynp', 'binseg', 'window', 'bottomup'
)
```

### Ensemble Method

Combines multiple detection methods for more robust regime identification.

```python
from src.regime_detection import EnsembleRegimeDetector, RegimeDetectorType

# Parameters
detector = EnsembleRegimeDetector(
    detectors=[                              # List of detector types to use
        RegimeDetectorType.KMEANS,
        RegimeDetectorType.HMM,
        RegimeDetectorType.RUPTURES
    ],
    weights={                                # Optional weights for each detector
        RegimeDetectorType.KMEANS: 1.0,
        RegimeDetectorType.HMM: 1.0,
        RegimeDetectorType.RUPTURES: 1.0
    }
)
```

## Common Interface

All detectors share a common interface:

### `fit(df)`

Fits the detector to the provided OHLCV dataframe and returns a dataframe with regime labels.

### `predict(df)`

Predicts regimes for new data and returns a dataframe with regime labels.

### `create_regime_dashboard(df)`

Creates an interactive dashboard visualizing the regimes and their characteristics, returning the path to the saved HTML file.

## Regime Types

The library identifies four common regime types:

1. **Trending**: Directional price movement with strong momentum
2. **Mean-reverting**: Range-bound market with price oscillations
3. **Volatile**: High uncertainty with significant price swings
4. **Breakout**: Sudden change in market direction or volatility

## Example

See the `regime_detection_example.py` file for a complete working example.

## Dashboards

Each detector can generate an interactive dashboard with visualizations including:
- Price charts with regime background colors
- Regime transition probability heatmaps
- Performance metrics by regime
- Feature importance charts
- Regime characteristics
- Volatility and return distributions

## Customization

You can customize detector parameters to optimize for your specific market or timeframe:

- Adjust the number of regimes
- Change feature calculation periods
- Modify regime classification thresholds
- Tune algorithm-specific parameters

## Performance Tips

- For large datasets, HMM and Bayesian methods can be computation-intensive
- Ruptures provides the fastest changepoint detection
- Consider using the EnsembleRegimeDetector with a subset of methods for balance of accuracy and performance
- For real-time applications, pre-fit the models and use the `predict()` method for new data 