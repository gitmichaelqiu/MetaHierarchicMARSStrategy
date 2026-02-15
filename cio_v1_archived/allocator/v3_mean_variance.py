"""
Allocator V3: Mean-Variance Optimization.

Type: allocator | Version: v3

Uses simplified Markowitz optimization: maximize expected return for
a given risk budget, using base model signals as return estimates.

    max  w'μ  -  λ * w'Σw
    s.t. Σw ≤ max_exposure, w_i ≥ 0
"""

import numpy as np
from typing import Dict
from allocator.base import BaseAllocator, AllocationOutput
from base_model_adapter import BaseModelSignal


class MeanVarianceAllocator(BaseAllocator):
    """v3: Mean-Variance optimization using model signals as views."""
    
    @property
    def version(self) -> str:
        return "v3-mean-variance"
    
    def __init__(
        self,
        risk_aversion: float = 2.0,
        min_weight: float = 0.02,
        max_weight: float = 0.35,
        lookback: int = 60,
    ):
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.lookback = lookback
        self._return_history: Dict[str, list] = {}
    
    def update_returns(self, daily_returns: Dict[str, float]):
        """Call daily to build covariance matrix."""
        for t, r in daily_returns.items():
            if t not in self._return_history:
                self._return_history[t] = []
            self._return_history[t].append(r)
            # Keep bounded
            if len(self._return_history[t]) > self.lookback * 2:
                self._return_history[t] = self._return_history[t][-self.lookback:]
    
    def allocate(
        self,
        signals: Dict[str, BaseModelSignal],
        max_exposure: float = 1.0,
    ) -> AllocationOutput:
        if not signals:
            return AllocationOutput(weights={}, cash_weight=1.0)
        
        tickers = list(signals.keys())
        n = len(tickers)
        
        # Expected returns from model signals
        # Signal = position * confidence, scaled to daily
        mu = np.array([
            signals[t].position * signals[t].confidence * 0.001
            for t in tickers
        ])
        
        # Covariance matrix from historical returns
        has_history = all(
            t in self._return_history and len(self._return_history[t]) >= 20
            for t in tickers
        )
        
        if has_history:
            ret_matrix = np.array([
                self._return_history[t][-self.lookback:]
                for t in tickers
            ])
            # Pad if lengths differ
            min_len = min(len(r) for r in ret_matrix)
            ret_matrix = np.array([r[-min_len:] for r in ret_matrix])
            
            Sigma = np.cov(ret_matrix)
            if Sigma.ndim == 0:
                Sigma = np.array([[Sigma]])
            # Regularize
            Sigma += np.eye(n) * 1e-6
        else:
            # Fallback: diagonal from individual volatilities
            vols = np.array([max(signals[t].volatility, 0.05) / np.sqrt(252) for t in tickers])
            Sigma = np.diag(vols ** 2)
        
        # Analytical solution: w* = (1/λ) * Σ^{-1} * μ
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            raw_w = (1.0 / self.risk_aversion) * Sigma_inv @ mu
        except np.linalg.LinAlgError:
            # Fallback to equal weight
            raw_w = np.ones(n) / n
        
        # Project to feasible region: w_i >= 0, sum <= max_exposure
        raw_w = np.maximum(raw_w, 0)  # Long only
        
        # Apply per-ticker constraints
        raw_w = np.minimum(raw_w, self.max_weight)
        raw_w = np.where(raw_w < self.min_weight, 0, raw_w)  # Remove tiny weights
        
        # Normalize to max_exposure
        w_sum = raw_w.sum()
        if w_sum > 0:
            if w_sum > max_exposure:
                raw_w = raw_w / w_sum * max_exposure
        
        weights = {tickers[i]: float(raw_w[i]) for i in range(n)}
        cash = 1.0 - sum(weights.values())
        
        return AllocationOutput(
            weights=weights,
            cash_weight=max(0, cash),
            metadata={
                'mu': {tickers[i]: float(mu[i]) for i in range(n)},
                'has_cov_history': has_history,
            }
        )
