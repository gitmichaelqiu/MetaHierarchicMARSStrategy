"""
Base Model Adapter — Standard interface for plugging base models into the CIO.

Any trading model (v6, v7, future versions) must be wrapped in an adapter
that implements the BaseModelAdapter protocol. This produces BaseModelSignal
objects that the CIO Allocator and Risk Manager consume.

See BASE_MODEL_GUIDE.md for how to create your own adapter.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field


@dataclass
class BaseModelSignal:
    """Standardized output from any base model."""
    ticker: str
    position: float          # Target position from model (-0.3 to 1.3)
    confidence: float        # Model confidence (0.0 to 1.0)
    regime: str              # Regime name: Growth/Stagnation/Transition/Crisis
    regime_probs: Dict[str, float]  # Full regime probabilities
    volatility: float        # Current rolling volatility
    metadata: Dict = field(default_factory=dict)  # baseline, overlay, drawdown, etc.


@runtime_checkable
class BaseModelAdapter(Protocol):
    """Protocol that all base model adapters must implement."""
    
    @property
    def model_name(self) -> str:
        """Human-readable model name, e.g. 'V7-CrossAsset'."""
        ...
    
    def initialize(self, ticker: str, df: pd.DataFrame, **kwargs) -> None:
        """
        Initialize the model for a specific ticker.
        Called once before backtesting begins.
        
        Args:
            ticker: Stock ticker symbol
            df: Full historical DataFrame with indicators
            **kwargs: Additional data (vix_data, weekly_trend, etc.)
        """
        ...
    
    def get_signal(self, df: pd.DataFrame, idx: int) -> BaseModelSignal:
        """
        Get the model's signal for the current bar.
        
        Args:
            df: Full historical DataFrame
            idx: Current bar index
            
        Returns:
            BaseModelSignal with position, confidence, regime
        """
        ...
    
    def reset(self) -> None:
        """Reset model state for a new backtest run."""
        ...
