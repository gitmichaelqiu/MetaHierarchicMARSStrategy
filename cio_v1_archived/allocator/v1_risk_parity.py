"""
Allocator V1: Risk Parity (Inverse Volatility Weighting).

Type: allocator | Version: v1

Allocates capital inversely proportional to each ticker's recent volatility.
Higher volatility → lower weight. This equalizes risk contribution.

    weight_i = (1 / vol_i) / Σ(1 / vol_j)
"""

import numpy as np
from typing import Dict
from allocator.base import BaseAllocator, AllocationOutput
from base_model_adapter import BaseModelSignal


class RiskParityAllocator(BaseAllocator):
    """v1: Inverse-volatility risk parity allocation."""
    
    @property
    def version(self) -> str:
        return "v1-risk-parity"
    
    def __init__(self, min_weight: float = 0.05, max_weight: float = 0.40):
        """
        Args:
            min_weight: Minimum allocation per ticker (prevents zero exposure)
            max_weight: Maximum allocation per ticker (prevents concentration)
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def allocate(
        self,
        signals: Dict[str, BaseModelSignal],
        max_exposure: float = 1.0,
    ) -> AllocationOutput:
        if not signals:
            return AllocationOutput(weights={}, cash_weight=1.0)
        
        # Compute inverse-vol weights
        inv_vols = {}
        for ticker, sig in signals.items():
            vol = max(sig.volatility, 0.05)  # Floor at 5% to avoid div by zero
            inv_vols[ticker] = 1.0 / vol
        
        total_inv_vol = sum(inv_vols.values())
        
        # Normalize
        raw_weights = {t: iv / total_inv_vol for t, iv in inv_vols.items()}
        
        # Apply min/max constraints
        weights = {}
        for ticker, w in raw_weights.items():
            weights[ticker] = np.clip(w, self.min_weight, self.max_weight)
        
        # Renormalize to sum to max_exposure
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {t: w / w_sum * max_exposure for t, w in weights.items()}
        
        cash = 1.0 - sum(weights.values())
        
        return AllocationOutput(
            weights=weights,
            cash_weight=max(0, cash),
            metadata={
                'raw_weights': raw_weights,
                'inv_vols': inv_vols,
            }
        )
