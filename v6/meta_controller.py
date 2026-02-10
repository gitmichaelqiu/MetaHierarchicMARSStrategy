"""
V6 Meta-Controller — Weekly Trend Confirmation + V5 Defense Layers.

V6 adds weekly EMA trend as a 4th defense layer:
1. Regime momentum (V5)
2. VIX fear gauge (V5)
3. Drawdown-adaptive baseline (V5)
4. **Weekly trend confirmation (V6 NEW)**

Weekly 13/26 EMA crossover gives 2-3 week early warning of trend reversals.
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
    V6 Meta-Controller with multi-timeframe trend confirmation.
    
    Four defense/boost layers:
    1. Regime momentum: detect transitions 3-5 days early
    2. VIX: market-wide fear gauge
    3. Drawdown-adaptive: scale baseline during losses
    4. Weekly trend: 13/26 EMA crossover for structural trend direction
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        min_trade_size: float = 0.02,
        # Regime momentum params
        regime_momentum_window: int = 5,
        regime_momentum_threshold: float = 0.12,
        # VIX params
        vix_caution_level: float = 25.0,
        vix_fear_level: float = 35.0,
        # Weekly trend params
        weekly_bullish_boost: float = 1.15,
        weekly_bearish_cut: float = 0.70,
        weekly_trend_threshold: float = 0.02,
    ):
        self.transaction_cost = transaction_cost
        self.min_trade_size = min_trade_size
        self.regime_momentum_threshold = regime_momentum_threshold
        self.vix_caution_level = vix_caution_level
        self.vix_fear_level = vix_fear_level
        self.weekly_bullish_boost = weekly_bullish_boost
        self.weekly_bearish_cut = weekly_bearish_cut
        self.weekly_trend_threshold = weekly_trend_threshold
        
        self.current_position: float = 0.0
        self.equity_curve: list = [1.0]
        self.peak_equity: float = 1.0
        
        self._growth_prob_history: deque = deque(maxlen=10)
        self._crisis_prob_history: deque = deque(maxlen=10)
    
    def _compute_baseline(self, regime_probs: Dict[str, float]) -> float:
        baseline = 0.0
        for regime, prob in regime_probs.items():
            baseline += prob * REGIME_BASELINES.get(regime, 0.3)
        return baseline
    
    def _compute_regime_momentum(self, regime_probs: Dict[str, float]) -> Dict[str, float]:
        growth_prob = regime_probs.get('Growth', 0)
        crisis_prob = regime_probs.get('Crisis', 0)
        self._growth_prob_history.append(growth_prob)
        self._crisis_prob_history.append(crisis_prob)
        
        momentum = {'growth': 0.0, 'crisis': 0.0}
        if len(self._growth_prob_history) >= 3:
            recent = list(self._growth_prob_history)
            current = np.mean(recent[-2:])
            earlier = np.mean(recent[:-2]) if len(recent) > 2 else recent[0]
            momentum['growth'] = current - earlier
        if len(self._crisis_prob_history) >= 3:
            recent = list(self._crisis_prob_history)
            current = np.mean(recent[-2:])
            earlier = np.mean(recent[:-2]) if len(recent) > 2 else recent[0]
            momentum['crisis'] = current - earlier
        return momentum
    
    def _apply_regime_momentum_adjustment(self, baseline: float, momentum: Dict[str, float]) -> float:
        adjusted = baseline
        if momentum['growth'] < -self.regime_momentum_threshold:
            fade_factor = min(0.4, abs(momentum['growth']) * 2)
            adjusted *= (1 - fade_factor)
        elif momentum['growth'] > self.regime_momentum_threshold:
            boost = min(0.15, momentum['growth'] * 0.8)
            adjusted = min(MAX_POSITION, adjusted + boost)
        if momentum['crisis'] > self.regime_momentum_threshold:
            cut_factor = min(0.5, momentum['crisis'] * 3)
            adjusted *= (1 - cut_factor)
        return adjusted
    
    def _apply_vix_adjustment(self, baseline: float, vix_value: Optional[float]) -> float:
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
    
    def _apply_weekly_trend_adjustment(self, baseline: float, weekly_trend: Optional[float]) -> float:
        """
        V6 NEW: Adjust baseline based on weekly 13/26 EMA trend.
        
        weekly_trend = (ema13 - ema26) / ema26
        Positive = weekly uptrend, Negative = weekly downtrend.
        
        This gives 2-3 week early warning of structural trend changes
        that daily regime detection misses.
        """
        if weekly_trend is None or np.isnan(weekly_trend):
            return baseline
        
        if weekly_trend > self.weekly_trend_threshold:
            # Weekly bullish — boost baseline
            # Scale boost by trend strength (capped at 1.15)
            strength = min(1.0, weekly_trend / 0.08)  # Full boost at 8% gap
            boost = 1.0 + (self.weekly_bullish_boost - 1.0) * strength
            return min(MAX_POSITION, baseline * boost)
        elif weekly_trend < -self.weekly_trend_threshold:
            # Weekly bearish — cut baseline
            # Scale cut by trend strength (capped at 0.70)
            strength = min(1.0, abs(weekly_trend) / 0.08)
            cut = 1.0 - (1.0 - self.weekly_bearish_cut) * strength
            return baseline * max(self.weekly_bearish_cut, cut)
        
        return baseline
    
    def _compute_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        current = self.equity_curve[-1]
        self.peak_equity = max(self.peak_equity, current)
        return max(0, 1 - current / self.peak_equity)
    
    def _apply_drawdown_scaling(self, baseline: float, drawdown: float) -> float:
        if drawdown > 0.25:
            return baseline * 0.4
        elif drawdown > 0.15:
            factor = 1.0 - (drawdown - 0.15) * 6
            return baseline * max(0.4, factor)
        elif drawdown > 0.08:
            factor = 1.0 - (drawdown - 0.08) * 2
            return baseline * max(0.85, factor)
        return baseline
    
    def _compute_risk_budget(self, drawdown: float) -> float:
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
        weekly_trend: Optional[float] = None,
    ) -> ControllerOutput:
        """
        V6 position computation: 4 defense/boost layers + overlay.
        
        Pipeline:
        1. Raw baseline from regime probs
        2. Regime momentum (early transition)
        3. Weekly trend confirmation (structural direction)
        4. VIX fear gauge
        5. Drawdown scaling
        6. Alpha overlay
        """
        # 1. Raw baseline
        raw_baseline = self._compute_baseline(regime_probs)
        
        # 2. Regime momentum
        momentum = self._compute_regime_momentum(regime_probs)
        baseline = self._apply_regime_momentum_adjustment(raw_baseline, momentum)
        
        # 3. Weekly trend confirmation (V6 NEW)
        baseline = self._apply_weekly_trend_adjustment(baseline, weekly_trend)
        
        # 4. VIX
        baseline = self._apply_vix_adjustment(baseline, vix_value)
        
        # 5. Drawdown scaling
        drawdown = self._compute_drawdown()
        baseline = self._apply_drawdown_scaling(baseline, drawdown)
        
        # 6. Alpha overlay
        risk_budget = self._compute_risk_budget(drawdown)
        overlay = action * confidence * OVERLAY_SCALE * risk_budget
        if overlay > 0:
            overlay *= 1.1
        else:
            overlay *= 0.9
        overlay = np.clip(overlay, -OVERLAY_SCALE, OVERLAY_SCALE)
        
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
                'weekly_trend': weekly_trend,
                'vix_value': vix_value,
                'drawdown': drawdown,
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
