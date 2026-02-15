"""
CIO v2 Risk Manager — Minimal, VIX-only emergency protection.

The base model IS the risk manager. This layer only protects against
extreme market events (VIX > 35) that the base model can't see.

Changes vs v1:
  - Removed: drawdown guard (base model already reduces position on DD)
  - Removed: correlation monitor (redundant, model has crisis agent)
  - Removed: regime-based cash (base model already tracks regimes)
  - Kept: VIX extreme fear (>35) as the ONE safety valve
  - Reduced: min_cash from 5% to 2%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from base_model_adapter import BaseModelSignal


@dataclass
class RiskOutput:
    """Risk manager output — constraints on portfolio allocation."""
    max_exposure: float        # Maximum invested fraction (1.0 - min_cash)
    position_scalar: float     # Multiplier on all positions
    risk_flags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class RiskManager:
    """
    v2: Minimal risk layer — only extreme VIX protection.
    The base model handles drawdowns, regimes, and correlations.
    """
    
    def __init__(
        self,
        min_cash: float = 0.02,       # Minimal cash buffer
        vix_extreme: float = 35.0,    # Only VIX > 35 triggers protection
        vix_panic: float = 45.0,      # Severe panic threshold
    ):
        self.min_cash = min_cash
        self.vix_extreme = vix_extreme
        self.vix_panic = vix_panic
        
        # State (kept for compatibility)
        self._equity_curve: list = [1.0]
        self._peak_equity: float = 1.0
    
    def update_equity(self, portfolio_return: float):
        new_eq = self._equity_curve[-1] * (1 + portfolio_return)
        self._equity_curve.append(new_eq)
        self._peak_equity = max(self._peak_equity, new_eq)
    
    @property
    def current_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        return max(0, 1 - self._equity_curve[-1] / self._peak_equity)
    
    def assess_risk(
        self,
        signals: Dict[str, BaseModelSignal],
        vix_value: Optional[float] = None,
        avg_correlation: Optional[float] = None,
    ) -> RiskOutput:
        flags = []
        cash_target = self.min_cash
        scalar = 1.0
        
        # Only one layer: VIX extreme protection
        if vix_value is not None and not np.isnan(vix_value):
            if vix_value > self.vix_panic:
                scalar = 0.5
                cash_target = 0.30
                flags.append(f"VIX_PANIC ({vix_value:.1f})")
            elif vix_value > self.vix_extreme:
                # Gentle scaling: 0.8 at VIX=35, 0.5 at VIX=45
                t = (vix_value - self.vix_extreme) / (self.vix_panic - self.vix_extreme)
                scalar = 1.0 - 0.5 * t
                cash_target = max(cash_target, 0.10 + 0.20 * t)
                flags.append(f"VIX_EXTREME ({vix_value:.1f})")
        
        max_exposure = 1.0 - cash_target
        
        return RiskOutput(
            max_exposure=max_exposure,
            position_scalar=scalar,
            risk_flags=flags,
            metadata={
                'drawdown': self.current_drawdown,
                'cash_target': cash_target,
                'vix': vix_value,
            }
        )
    
    def reset(self):
        self._equity_curve = [1.0]
        self._peak_equity = 1.0
