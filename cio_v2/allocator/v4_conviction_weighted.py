"""
Allocator V4: Conviction-Weighted (Direct Trust).

Type: allocator | Version: v4

Weights tickers proportionally to model position^α.
No artificial cash reserve. Cash only when models go to zero.
The base model IS the risk manager — we just allocate capital.

    w_i = max(pos_i, 0)^α / Σ max(pos_j, 0)^α
"""

import numpy as np
from typing import Dict
from allocator.base import BaseAllocator, AllocationOutput
from base_model_adapter import BaseModelSignal


class ConvictionWeightedAllocator(BaseAllocator):
    """v4: Direct Trust — weight by model conviction, no artificial cash."""
    
    @property
    def version(self) -> str:
        return "v4-conviction-weighted"
    
    def __init__(
        self,
        alpha: float = 1.5,       # Conviction amplifier (1.0=linear, 2.0=quadratic)
        min_position: float = 0.05,  # Ignore signals below this
        max_weight: float = 0.40,   # Max per-ticker cap (safety only)
    ):
        self.alpha = alpha
        self.min_position = min_position
        self.max_weight = max_weight
    
    def allocate(
        self,
        signals: Dict[str, BaseModelSignal],
        max_exposure: float = 1.0,
    ) -> AllocationOutput:
        if not signals:
            return AllocationOutput(weights={}, cash_weight=1.0)
        
        tickers = list(signals.keys())
        
        # Raw conviction from model position
        raw = {}
        for t in tickers:
            pos = max(signals[t].position, 0.0)
            if pos < self.min_position:
                raw[t] = 0.0
            else:
                raw[t] = pos ** self.alpha
        
        total = sum(raw.values())
        
        if total <= 0:
            # All models say zero — go full cash
            return AllocationOutput(
                weights={t: 0.0 for t in tickers},
                cash_weight=1.0,
                metadata={'reason': 'all_models_zero'}
            )
        
        # Normalize to max_exposure
        weights = {}
        for t in tickers:
            w = (raw[t] / total) * max_exposure
            weights[t] = min(w, self.max_weight)
        
        # Re-normalize if max_weight clipped anything
        w_sum = sum(weights.values())
        if w_sum > max_exposure:
            scale = max_exposure / w_sum
            weights = {t: w * scale for t, w in weights.items()}
        
        cash = 1.0 - sum(weights.values())
        
        return AllocationOutput(
            weights=weights,
            cash_weight=max(0, cash),
            metadata={
                'raw_conviction': {t: float(raw[t]) for t in tickers},
                'alpha': self.alpha,
            }
        )
