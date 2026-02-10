"""
V3 Meta-Controller — Regime-aware position bias + asymmetric sizing.

Key V3 changes over V2:
- Bullish floor during Growth: min position = growth_prob × 0.6
- Neutral floor during Stagnation: maintain small long position (0.15) when no bearish signal
- Asymmetric sizing: longs get 1.3× boost, shorts get 0.7× reduction
- Removed extreme-volatility scaling (was dampening NVDA/TSLA)
- Risk budget floor remains 0.4
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
    metadata: Dict


class MetaController:
    """
    V3 Meta-Controller — Regime-aware sizing with structural long bias.
    
    Key innovation: position floors during Growth and neutral tilt
    during Stagnation eliminate the "near-zero for 60% of bars" problem.
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        min_trade_size: float = 0.02,
        switching_cost_lambda: float = 0.005,
        drawdown_lookback: int = 60,
        # V3 new parameters
        growth_floor_factor: float = 0.6,        # Growth min pos = prob × this
        stagnation_neutral_bias: float = 0.15,    # Small long tilt during Stagnation
        long_boost: float = 1.3,                  # Asymmetric: longs get boosted
        short_dampen: float = 0.7,                # Asymmetric: shorts get dampened
    ):
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.min_trade_size = min_trade_size
        self.switching_cost_lambda = switching_cost_lambda
        self.drawdown_lookback = drawdown_lookback
        self.growth_floor_factor = growth_floor_factor
        self.stagnation_neutral_bias = stagnation_neutral_bias
        self.long_boost = long_boost
        self.short_dampen = short_dampen
        
        # State
        self.current_position: float = 0.0
        self.equity_curve: list = [1.0]
        self.peak_equity: float = 1.0
        
    def _compute_risk_budget(self) -> float:
        """Drawdown-based risk budget with floor at 0.4."""
        if len(self.equity_curve) < 2:
            return 1.0
        
        current_equity = self.equity_curve[-1]
        lookback = self.equity_curve[-self.drawdown_lookback:]
        peak = max(lookback)
        self.peak_equity = max(self.peak_equity, current_equity)
        
        drawdown = 1 - (current_equity / self.peak_equity)
        
        if drawdown > 0.15:
            budget = max(0.4, 1.0 - drawdown * 3)
        elif drawdown > 0.05:
            budget = max(0.6, 1.0 - drawdown * 2)
        else:
            budget = 1.0
        
        return budget
    
    def _compute_regime_bias(
        self,
        action: float,
        regime_probs: Dict[str, float]
    ) -> float:
        """
        V3 NEW: Apply regime-aware position bias.
        
        - Growth → enforce bullish floor
        - Stagnation → add small long tilt if not explicitly bearish
        - Crisis → no bias (allow full short)
        """
        growth_prob = regime_probs.get('Growth', 0)
        stagnation_prob = regime_probs.get('Stagnation', 0)
        crisis_prob = regime_probs.get('Crisis', 0)
        
        biased_action = action
        
        # Growth bias: enforce minimum long position
        if growth_prob > 0.4:
            floor = growth_prob * self.growth_floor_factor
            if biased_action < floor:
                biased_action = max(biased_action, floor)
        
        # Stagnation bias: add small long tilt if action is near-zero
        if stagnation_prob > 0.4 and abs(action) < 0.1 and crisis_prob < 0.3:
            biased_action = biased_action + self.stagnation_neutral_bias
        
        return biased_action
    
    def _apply_asymmetric_sizing(self, position: float) -> float:
        """
        V3 NEW: Asymmetric position sizing.
        Longs get 1.3× boost, shorts get 0.7× reduction.
        Reflects equity market's structural upward bias.
        """
        if position > 0:
            return position * self.long_boost
        elif position < 0:
            return position * self.short_dampen
        return position
    
    def _compute_switching_cost(self, target: float) -> float:
        """Transaction cost penalty for position changes."""
        delta = abs(target - self.current_position)
        return delta * self.switching_cost_lambda
    
    def compute_position(
        self,
        action: float,
        confidence: float,
        current_volatility: float,
        regime_probs: Dict[str, float]
    ) -> ControllerOutput:
        """
        Compute target position with V3 regime-aware bias.
        
        Pipeline:
        1. Base position = sign(action) × |action| × confidence × risk_budget
        2. Apply regime bias (Growth floor, Stagnation tilt)
        3. Apply asymmetric sizing (boost longs, dampen shorts)
        4. Check switching cost and min trade size
        """
        
        risk_budget = self._compute_risk_budget()
        
        # V3: Apply regime bias to action BEFORE position sizing
        biased_action = self._compute_regime_bias(action, regime_probs)
        
        # Direct position sizing (same as V2 — no Kelly, no smoothing)
        base_position = np.sign(biased_action) * abs(biased_action) * confidence * risk_budget
        
        # V3: Asymmetric sizing
        sized_position = self._apply_asymmetric_sizing(base_position)
        
        # Clamp to max position
        target_position = np.clip(sized_position, -self.max_position, self.max_position)
        
        # Check if trade is worth executing
        delta = abs(target_position - self.current_position)
        switching_cost = self._compute_switching_cost(target_position)
        is_trade_allowed = delta >= self.min_trade_size
        
        return ControllerOutput(
            target_position=target_position,
            risk_budget=risk_budget,
            is_trade_allowed=is_trade_allowed,
            metadata={
                'action': action,
                'biased_action': biased_action,
                'confidence': confidence,
                'regime_probs': regime_probs,
                'switching_cost': switching_cost,
                'base_position': base_position,
                'pre_asymmetric': base_position,
                'post_asymmetric': sized_position,
            }
        )
    
    def execute_trade(self, target_position: float):
        """Execute position change."""
        self.current_position = target_position
    
    def update_equity(self, portfolio_return: float):
        """Update equity curve."""
        new_equity = self.equity_curve[-1] * (1 + portfolio_return)
        self.equity_curve.append(new_equity)
        self.peak_equity = max(self.peak_equity, new_equity)
    
    def reset(self):
        """Reset for new backtest."""
        self.current_position = 0.0
        self.equity_curve = [1.0]
        self.peak_equity = 1.0
