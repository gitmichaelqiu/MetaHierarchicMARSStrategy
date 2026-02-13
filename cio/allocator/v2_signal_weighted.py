"""
Allocator V2: Signal-Weighted Allocation.

Type: allocator | Version: v2

Combines risk parity with base model confidence and position direction.
Tickers with stronger signals and higher confidence get more capital.
Crisis-regime tickers get near-zero allocation.

    weight_i = (1 / vol_i) * confidence_i * max(position_i, 0)
"""

import numpy as np
from typing import Dict
from allocator.base import BaseAllocator, AllocationOutput
from base_model_adapter import BaseModelSignal


class SignalWeightedAllocator(BaseAllocator):
    """v2: Risk parity × model confidence × position direction."""
    
    @property
    def version(self) -> str:
        return "v2-signal-weighted"
    
    def __init__(
        self,
        min_weight: float = 0.03,
        max_weight: float = 0.45,
        crisis_penalty: float = 0.2,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.crisis_penalty = crisis_penalty
    
    def allocate(
        self,
        signals: Dict[str, BaseModelSignal],
        max_exposure: float = 1.0,
    ) -> AllocationOutput:
        if not signals:
            return AllocationOutput(weights={}, cash_weight=1.0)
        
        scores = {}
        for ticker, sig in signals.items():
            vol = max(sig.volatility, 0.05)
            inv_vol = 1.0 / vol
            
            # Confidence-adjusted position (only long side contributes)
            signal_strength = max(sig.position, 0.0) * sig.confidence
            
            # Regime penalty
            regime_mult = 1.0
            if sig.regime == 'Crisis':
                regime_mult = self.crisis_penalty
            elif sig.regime == 'Transition':
                regime_mult = 0.5
            
            scores[ticker] = inv_vol * (0.3 + 0.7 * signal_strength) * regime_mult
        
        total = sum(scores.values())
        if total <= 0:
            n = len(signals)
            weights = {t: max_exposure / n for t in signals}
        else:
            raw = {t: s / total for t, s in scores.items()}
            weights = {t: np.clip(w, self.min_weight, self.max_weight) for t, w in raw.items()}
            w_sum = sum(weights.values())
            if w_sum > 0:
                weights = {t: w / w_sum * max_exposure for t, w in weights.items()}
        
        cash = 1.0 - sum(weights.values())
        
        return AllocationOutput(
            weights=weights,
            cash_weight=max(0, cash),
            metadata={'scores': scores}
        )
