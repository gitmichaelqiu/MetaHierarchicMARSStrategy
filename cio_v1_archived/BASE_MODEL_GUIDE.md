# How to Add Your Base Model to the CIO Framework

## Overview

The CIO Framework uses **adapters** to connect base models (v6, v7, etc.) to the portfolio allocation engine. Each adapter wraps a trading model into a standard interface that produces `BaseModelSignal` objects.

## Step 1: Implement the Adapter

Create a new file in `cio/adapters/` (e.g., `my_model_adapter.py`):

```python
from base_model_adapter import BaseModelAdapter, BaseModelSignal

class MyModelAdapter:
    """Wraps MyModel into the CIO BaseModelAdapter protocol."""
    
    @property
    def model_name(self) -> str:
        return "MyModel-v1"
    
    def initialize(self, ticker: str, df: pd.DataFrame, **kwargs) -> None:
        """Set up model state. Called once per ticker."""
        self._ticker = ticker
        # Initialize your model components here
        # kwargs may include: vix_data, weekly_trend, sector_trend, etc.
    
    def get_signal(self, df: pd.DataFrame, idx: int) -> BaseModelSignal:
        """Return signal for bar idx."""
        # Run your model logic here
        return BaseModelSignal(
            ticker=self._ticker,
            position=0.5,        # Your model's target position (-0.3 to 1.3)
            confidence=0.8,      # How confident (0.0 to 1.0)
            regime='Growth',     # Regime classification
            regime_probs={'Growth': 0.7, 'Stagnation': 0.3},
            volatility=0.20,     # Current rolling volatility
            metadata={}          # Any extra info
        )
    
    def reset(self) -> None:
        """Reset state for new backtest run."""
        pass
```

## Step 2: Register in `run_benchmark.py`

Import your adapter and use it:

```python
from adapters.my_model_adapter import MyModelAdapter

# In the adapter initialization loop:
adapter = MyModelAdapter()
adapter.initialize(ticker, ticker_data[ticker], **kwargs)
```

## Key Requirements

| Field          | Type  | Range                               | Description                     |
| -------------- | ----- | ----------------------------------- | ------------------------------- |
| `position`     | float | -0.3 to 1.3                         | Target position from your model |
| `confidence`   | float | 0.0 to 1.0                          | Model confidence in signal      |
| `regime`       | str   | Growth/Stagnation/Transition/Crisis | Regime label                    |
| `regime_probs` | dict  | Probabilities sum to 1.0            | Full regime distribution        |
| `volatility`   | float | 0.01+                               | Rolling annualized volatility   |

## Reference

See `cio/adapters/v7_adapter.py` for the full reference implementation.
