"""
Allocater V5: Absolute Conviction (True Direct Trust).

Type: allocator | Version: v5

Weights tickers conditionally allowing cash to build up dynamically.
Unlike V4 which normalizes weights to always fully invest the portfolio,
V5 preserves the absolute magnitude of the base models' positions.
If models scale down due to risk, the portfolio scales down naturally into cash.
It also supports negative (short) allocations.
"""

import numpy as np
from typing import Dict
from allocator.base import BaseAllocator, AllocationOutput
from base_model_adapter import BaseModelSignal


class AbsoluteConvictionAllocator(BaseAllocator):
    """
    v5: True Direct Trust — absolute conviction, allows shorts and dynamic cash.
    Does NOT normalize (divide by sum) unless the sum exceeds max_exposure.
    """
    
    @property
    def version(self) -> str:
        return "v5-absolute-conviction"
    
    def __init__(
        self,
        alpha: float = 1.0,          # Conviction amplifier (1.0=linear)
        min_position: float = 0.05,  # Ignore signals strictly between (-min, +min)
        max_weight: float = 0.40,    # Max absolute per-ticker cap
        allow_shorts: bool = True,   # Allow negative weights
    ):
        self.alpha = alpha
        self.min_position = min_position
        self.max_weight = max_weight
        self.allow_shorts = allow_shorts
    
    def allocate(
        self,
        signals: Dict[str, BaseModelSignal],
        max_exposure: float = 1.0,
    ) -> AllocationOutput:
        if not signals:
            return AllocationOutput(weights={}, cash_weight=1.0)
        
        tickers = list(signals.keys())
        
        # Calculate raw weights preserving sign
        raw_weights = {}
        for t in tickers:
            pos = signals[t].position
            
            if not self.allow_shorts:
                pos = max(0.0, pos)
                
            abs_pos = abs(pos)
            
            if abs_pos < self.min_position:
                raw_weights[t] = 0.0
            else:
                # Retain sign, apply alpha to magnitude
                sign = 1.0 if pos > 0 else -1.0
                raw_weights[t] = sign * (abs_pos ** self.alpha)
        
        # Apply per-ticker caps FIRST
        capped_weights = {}
        for t, w in raw_weights.items():
            if w > 0:
                capped_weights[t] = min(w, self.max_weight)
            elif w < 0:
                capped_weights[t] = max(w, -self.max_weight)
            else:
                capped_weights[t] = 0.0
                
        # Now check if the GROSS exposure exceeds max_exposure
        gross_exposure = sum(abs(w) for w in capped_weights.values())
        
        final_weights = {}
        if gross_exposure > max_exposure:
            # We must normalize down to fit within the risk limits
            scale = max_exposure / gross_exposure
            for t, w in capped_weights.items():
                final_weights[t] = w * scale
        else:
            # GROSS exposure is less than max_exposure.
            # Do NOT normalize upwards. This is the key difference from V4.
            final_weights = capped_weights.copy()
            
        # In this allocator, cash is whatever is NOT held in gross exposure limits
        final_gross = sum(abs(w) for w in final_weights.values())
        cash = 1.0 - final_gross
        
        return AllocationOutput(
            weights=final_weights,
            cash_weight=max(0.0, cash),
            metadata={
                'raw_conviction': {t: float(raw_weights[t]) for t in tickers},
                'gross_exposure': float(final_gross),
                'alpha': self.alpha,
                'normalized': gross_exposure > max_exposure
            }
        )
