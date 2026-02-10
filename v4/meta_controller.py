"""
V4 Meta-Controller — Adaptive Baseline + Alpha Overlay.

The fundamental architectural shift: instead of starting from 0.0 (flat)
and using signals to go long, start from a regime-dependent baseline
(always long) and use MoA signals as an adjustment overlay.

position = baseline(regime_probs) + overlay(moa_signal) × scale
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ControllerOutput:
    """Output from the meta-controller."""
    target_position: float
    risk_budget: float
    is_trade_allowed: bool
    baseline: float
    overlay: float
    metadata: Dict


# Regime baselines — the core of V4
REGIME_BASELINES = {
    'Growth':     0.80,   # Nearly fully long
    'Stagnation': 0.50,   # Half long — capture equity drift
    'Transition': 0.40,   # Cautious long
    'Crisis':     0.10,   # Minimal exposure
}

OVERLAY_SCALE = 0.4       # MoA signal adjusts position by ±0.4
MAX_POSITION = 1.3        # Allow slight overweight
MIN_POSITION = -0.3       # Allow modest short
LONG_BOOST = 1.1          # Mild asymmetric boost for longs
SHORT_DAMPEN = 0.9        # Mild asymmetric dampen for shorts


class MetaController:
    """
    V4 Adaptive Baseline Meta-Controller.
    
    Key insight: B&H holds 100% long forever. To compete, the default
    position should be long, not flat. MoA signals add alpha on top.
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        min_trade_size: float = 0.02,
        switching_cost_lambda: float = 0.003,
        drawdown_lookback: int = 60,
    ):
        self.transaction_cost = transaction_cost
        self.min_trade_size = min_trade_size
        self.switching_cost_lambda = switching_cost_lambda
        self.drawdown_lookback = drawdown_lookback
        
        self.current_position: float = 0.0
        self.equity_curve: list = [1.0]
        self.peak_equity: float = 1.0
    
    def _compute_baseline(self, regime_probs: Dict[str, float]) -> float:
        """
        Probability-weighted baseline position.
        Smooth blending — no discrete regime switching.
        
        baseline = Σ (regime_prob × regime_baseline)
        """
        baseline = 0.0
        for regime, prob in regime_probs.items():
            baseline += prob * REGIME_BASELINES.get(regime, 0.3)
        return baseline
    
    def _compute_risk_budget(self) -> float:
        """Drawdown-based risk budget. Reduces overlay, not baseline."""
        if len(self.equity_curve) < 2:
            return 1.0
        
        current_equity = self.equity_curve[-1]
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = 1 - (current_equity / self.peak_equity)
        
        if drawdown > 0.15:
            return max(0.3, 1.0 - drawdown * 3)
        elif drawdown > 0.05:
            return max(0.5, 1.0 - drawdown * 2)
        return 1.0
    
    def _compute_overlay(
        self,
        action: float,
        confidence: float,
        risk_budget: float,
    ) -> float:
        """
        Compute the alpha overlay from MoA signals.
        Overlay adjusts position by ±OVERLAY_SCALE around the baseline.
        """
        raw_overlay = action * confidence * OVERLAY_SCALE
        
        # Risk budget only affects the overlay, not the baseline
        scaled_overlay = raw_overlay * risk_budget
        
        # Asymmetric: boost positive overlays slightly
        if scaled_overlay > 0:
            scaled_overlay *= LONG_BOOST
        else:
            scaled_overlay *= SHORT_DAMPEN
        
        return np.clip(scaled_overlay, -OVERLAY_SCALE, OVERLAY_SCALE)
    
    def compute_position(
        self,
        action: float,
        confidence: float,
        current_volatility: float,
        regime_probs: Dict[str, float]
    ) -> ControllerOutput:
        """
        Compute target position = baseline + overlay.
        """
        # Baseline from regime probabilities
        baseline = self._compute_baseline(regime_probs)
        
        # Risk budget (affects overlay only)
        risk_budget = self._compute_risk_budget()
        
        # Alpha overlay from MoA
        overlay = self._compute_overlay(action, confidence, risk_budget)
        
        # Final position
        target = baseline + overlay
        target = np.clip(target, MIN_POSITION, MAX_POSITION)
        
        # Check if trade is worth executing
        delta = abs(target - self.current_position)
        is_trade_allowed = delta >= self.min_trade_size
        
        return ControllerOutput(
            target_position=target,
            risk_budget=risk_budget,
            is_trade_allowed=is_trade_allowed,
            baseline=baseline,
            overlay=overlay,
            metadata={
                'action': action,
                'confidence': confidence,
                'regime_probs': regime_probs,
            }
        )
    
    def execute_trade(self, target_position: float):
        self.current_position = target_position
    
    def update_equity(self, portfolio_return: float):
        new_equity = self.equity_curve[-1] * (1 + portfolio_return)
        self.equity_curve.append(new_equity)
        self.peak_equity = max(self.peak_equity, new_equity)
    
    def reset(self):
        self.current_position = 0.0
        self.equity_curve = [1.0]
        self.peak_equity = 1.0
