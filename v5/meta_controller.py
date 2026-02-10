"""
V5 Meta-Controller — Regime Momentum + VIX Fear Gauge + Drawdown-Adaptive Baseline.

Three key improvements over V4:
1. Regime momentum: tracks probability direction to detect transitions early
2. VIX fear gauge: market-wide risk signal
3. Drawdown-adaptive baseline: scales baseline down during drawdowns (not just overlay)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class ControllerOutput:
    target_position: float
    risk_budget: float
    is_trade_allowed: bool
    baseline: float
    overlay: float
    metadata: Dict


# Regime baselines (same as V4)
REGIME_BASELINES = {
    'Growth':     0.80,
    'Stagnation': 0.50,
    'Transition': 0.40,
    'Crisis':     0.10,
}

OVERLAY_SCALE = 0.4
MAX_POSITION = 1.3
MIN_POSITION = -0.3


class MetaController:
    """
    V5 Meta-Controller with early-warning systems.
    
    Three layers of defense:
    1. Regime momentum: detect transitions 3-5 days early
    2. VIX: market-wide fear gauge
    3. Drawdown-adaptive: scale everything down during losses
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        min_trade_size: float = 0.02,
        drawdown_lookback: int = 60,
        # Regime momentum params
        regime_momentum_window: int = 5,
        regime_momentum_threshold: float = 0.12,
        # VIX params
        vix_caution_level: float = 25.0,
        vix_fear_level: float = 35.0,
    ):
        self.transaction_cost = transaction_cost
        self.min_trade_size = min_trade_size
        self.drawdown_lookback = drawdown_lookback
        self.regime_momentum_window = regime_momentum_window
        self.regime_momentum_threshold = regime_momentum_threshold
        self.vix_caution_level = vix_caution_level
        self.vix_fear_level = vix_fear_level
        
        self.current_position: float = 0.0
        self.equity_curve: list = [1.0]
        self.peak_equity: float = 1.0
        
        # V5: Regime probability history for momentum
        self._growth_prob_history: deque = deque(maxlen=10)
        self._crisis_prob_history: deque = deque(maxlen=10)
    
    def _compute_baseline(self, regime_probs: Dict[str, float]) -> float:
        """Probability-weighted baseline."""
        baseline = 0.0
        for regime, prob in regime_probs.items():
            baseline += prob * REGIME_BASELINES.get(regime, 0.3)
        return baseline
    
    def _compute_regime_momentum(self, regime_probs: Dict[str, float]) -> Dict[str, float]:
        """
        V5 NEW: Track regime probability direction.
        Returns momentum for growth and crisis.
        """
        growth_prob = regime_probs.get('Growth', 0)
        crisis_prob = regime_probs.get('Crisis', 0)
        
        self._growth_prob_history.append(growth_prob)
        self._crisis_prob_history.append(crisis_prob)
        
        momentum = {'growth': 0.0, 'crisis': 0.0}
        
        if len(self._growth_prob_history) >= 3:
            recent = list(self._growth_prob_history)
            # Compare current vs average of earlier entries
            current = np.mean(recent[-2:])
            earlier = np.mean(recent[:-2]) if len(recent) > 2 else recent[0]
            momentum['growth'] = current - earlier
        
        if len(self._crisis_prob_history) >= 3:
            recent = list(self._crisis_prob_history)
            current = np.mean(recent[-2:])
            earlier = np.mean(recent[:-2]) if len(recent) > 2 else recent[0]
            momentum['crisis'] = current - earlier
        
        return momentum
    
    def _apply_regime_momentum_adjustment(
        self, baseline: float, momentum: Dict[str, float]
    ) -> float:
        """
        V5 NEW: Adjust baseline based on regime probability direction.
        - Growth prob rising fast → boost baseline
        - Growth prob falling fast → reduce baseline  
        - Crisis prob rising fast → reduce baseline aggressively
        """
        adjusted = baseline
        
        # Growth momentum
        if momentum['growth'] < -self.regime_momentum_threshold:
            # Growth fading — reduce exposure proactively
            fade_factor = min(0.4, abs(momentum['growth']) * 2)
            adjusted *= (1 - fade_factor)
        elif momentum['growth'] > self.regime_momentum_threshold:
            # Growth strengthening — boost slightly
            boost = min(0.15, momentum['growth'] * 0.8)
            adjusted = min(MAX_POSITION, adjusted + boost)
        
        # Crisis momentum
        if momentum['crisis'] > self.regime_momentum_threshold:
            # Crisis building — cut exposure early
            cut_factor = min(0.5, momentum['crisis'] * 3)
            adjusted *= (1 - cut_factor)
        
        return adjusted
    
    def _apply_vix_adjustment(self, baseline: float, vix_value: Optional[float]) -> float:
        """
        V5 NEW: Adjust baseline based on VIX level.
        VIX > 25: reduce 20%, VIX > 35: reduce 50%
        VIX < 15: boost 10%
        """
        if vix_value is None or np.isnan(vix_value):
            return baseline
        
        if vix_value > self.vix_fear_level:
            return baseline * 0.5
        elif vix_value > self.vix_caution_level:
            factor = 1.0 - 0.2 * (vix_value - self.vix_caution_level) / (self.vix_fear_level - self.vix_caution_level)
            return baseline * max(0.5, factor)
        elif vix_value < 15:
            return min(MAX_POSITION, baseline * 1.1)
        
        return baseline
    
    def _compute_drawdown(self) -> float:
        """Current drawdown from peak."""
        if len(self.equity_curve) < 2:
            return 0.0
        current = self.equity_curve[-1]
        self.peak_equity = max(self.peak_equity, current)
        return max(0, 1 - current / self.peak_equity)
    
    def _apply_drawdown_scaling(self, baseline: float, drawdown: float) -> float:
        """
        V5 MODIFIED: Scale baseline down during drawdowns (not just overlay).
        
        Softened thresholds to avoid over-cutting during volatile recoveries.
        TSLA-type stocks need room to recover after 15-20% drawdowns.
        """
        if drawdown > 0.25:
            return baseline * 0.4
        elif drawdown > 0.15:
            # Gradual: 1.0 at 15% → 0.4 at 25%
            factor = 1.0 - (drawdown - 0.15) * 6
            return baseline * max(0.4, factor)
        elif drawdown > 0.08:
            # Mild: 1.0 at 8% → ~0.85 at 15%
            factor = 1.0 - (drawdown - 0.08) * 2
            return baseline * max(0.85, factor)
        return baseline
    
    def _compute_risk_budget(self, drawdown: float) -> float:
        """Risk budget for overlay (separate from baseline scaling)."""
        if drawdown > 0.15:
            return max(0.3, 1.0 - drawdown * 3)
        elif drawdown > 0.05:
            return max(0.5, 1.0 - drawdown * 2)
        return 1.0
    
    def compute_position(
        self,
        action: float,
        confidence: float,
        current_volatility: float,
        regime_probs: Dict[str, float],
        vix_value: Optional[float] = None,
    ) -> ControllerOutput:
        """
        V5 position computation with three layers of defense.
        
        Pipeline:
        1. Compute raw baseline from regime probs
        2. Adjust for regime momentum (early transition detection)
        3. Adjust for VIX (market-wide fear)
        4. Adjust for drawdown (capital preservation)
        5. Add alpha overlay
        """
        # 1. Raw baseline
        raw_baseline = self._compute_baseline(regime_probs)
        
        # 2. Regime momentum adjustment
        momentum = self._compute_regime_momentum(regime_probs)
        baseline = self._apply_regime_momentum_adjustment(raw_baseline, momentum)
        
        # 3. VIX adjustment
        baseline = self._apply_vix_adjustment(baseline, vix_value)
        
        # 4. Drawdown scaling (applies to baseline)
        drawdown = self._compute_drawdown()
        baseline = self._apply_drawdown_scaling(baseline, drawdown)
        
        # 5. Alpha overlay
        risk_budget = self._compute_risk_budget(drawdown)
        overlay = action * confidence * OVERLAY_SCALE * risk_budget
        
        # Mild asymmetric boost
        if overlay > 0:
            overlay *= 1.1
        else:
            overlay *= 0.9
        
        overlay = np.clip(overlay, -OVERLAY_SCALE, OVERLAY_SCALE)
        
        # Final position
        target = baseline + overlay
        target = np.clip(target, MIN_POSITION, MAX_POSITION)
        
        delta = abs(target - self.current_position)
        is_trade_allowed = delta >= self.min_trade_size
        
        return ControllerOutput(
            target_position=target,
            risk_budget=risk_budget,
            is_trade_allowed=is_trade_allowed,
            baseline=baseline,
            overlay=overlay,
            metadata={
                'raw_baseline': raw_baseline,
                'regime_momentum': momentum,
                'vix_value': vix_value,
                'drawdown': drawdown,
                'action': action,
                'confidence': confidence,
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
        self._growth_prob_history.clear()
        self._crisis_prob_history.clear()
