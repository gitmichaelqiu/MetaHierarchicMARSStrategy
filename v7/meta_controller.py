"""
V7 Meta-Controller — 6 Defense/Boost Layers.

1. Regime momentum (V5)
2. Weekly stock trend (V6)
3. Sector momentum — XLK trend confirmation (V7 NEW)
4. VIX fear gauge + term structure (V5 + V7 enhancement)
5. Correlation-based portfolio risk (V7 NEW)
6. Drawdown-adaptive baseline (V5)
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
    """V7 Meta-Controller with cross-asset intelligence."""
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        min_trade_size: float = 0.02,
        # Regime momentum
        regime_momentum_threshold: float = 0.12,
        # VIX
        vix_caution_level: float = 25.0,
        vix_fear_level: float = 35.0,
        # Weekly trend
        weekly_bullish_boost: float = 1.15,
        weekly_bearish_cut: float = 0.70,
        weekly_trend_threshold: float = 0.02,
        # V7: Sector momentum (boost-only, asymmetric)
        sector_confirm_boost: float = 1.12,
        # V7: Correlation risk (scales overlay, NOT baseline)
        corr_overlay_dampen: float = 0.70,
        corr_threshold: float = 0.70,
    ):
        self.transaction_cost = transaction_cost
        self.min_trade_size = min_trade_size
        self.regime_momentum_threshold = regime_momentum_threshold
        self.vix_caution_level = vix_caution_level
        self.vix_fear_level = vix_fear_level
        self.weekly_bullish_boost = weekly_bullish_boost
        self.weekly_bearish_cut = weekly_bearish_cut
        self.weekly_trend_threshold = weekly_trend_threshold
        self.sector_confirm_boost = sector_confirm_boost
        self.corr_overlay_dampen = corr_overlay_dampen
        self.corr_threshold = corr_threshold
        
        self.current_position: float = 0.0
        self.equity_curve: list = [1.0]
        self.peak_equity: float = 1.0
        self._growth_prob_history: deque = deque(maxlen=10)
        self._crisis_prob_history: deque = deque(maxlen=10)
    
    # ──── Layer 0: Baseline ────
    def _compute_baseline(self, regime_probs: Dict[str, float]) -> float:
        baseline = 0.0
        for regime, prob in regime_probs.items():
            baseline += prob * REGIME_BASELINES.get(regime, 0.3)
        return baseline
    
    # ──── Layer 1: Regime Momentum ────
    def _compute_regime_momentum(self, regime_probs: Dict[str, float]) -> Dict[str, float]:
        growth_prob = regime_probs.get('Growth', 0)
        crisis_prob = regime_probs.get('Crisis', 0)
        self._growth_prob_history.append(growth_prob)
        self._crisis_prob_history.append(crisis_prob)
        momentum = {'growth': 0.0, 'crisis': 0.0}
        if len(self._growth_prob_history) >= 3:
            recent = list(self._growth_prob_history)
            momentum['growth'] = np.mean(recent[-2:]) - (np.mean(recent[:-2]) if len(recent) > 2 else recent[0])
        if len(self._crisis_prob_history) >= 3:
            recent = list(self._crisis_prob_history)
            momentum['crisis'] = np.mean(recent[-2:]) - (np.mean(recent[:-2]) if len(recent) > 2 else recent[0])
        return momentum
    
    def _apply_regime_momentum(self, baseline: float, momentum: Dict[str, float]) -> float:
        adjusted = baseline
        if momentum['growth'] < -self.regime_momentum_threshold:
            adjusted *= (1 - min(0.4, abs(momentum['growth']) * 2))
        elif momentum['growth'] > self.regime_momentum_threshold:
            adjusted = min(MAX_POSITION, adjusted + min(0.15, momentum['growth'] * 0.8))
        if momentum['crisis'] > self.regime_momentum_threshold:
            adjusted *= (1 - min(0.5, momentum['crisis'] * 3))
        return adjusted
    
    # ──── Layer 2: Weekly Stock Trend ────
    def _apply_weekly_trend(self, baseline: float, weekly_trend: Optional[float]) -> float:
        if weekly_trend is None or np.isnan(weekly_trend):
            return baseline
        if weekly_trend > self.weekly_trend_threshold:
            strength = min(1.0, weekly_trend / 0.08)
            boost = 1.0 + (self.weekly_bullish_boost - 1.0) * strength
            return min(MAX_POSITION, baseline * boost)
        elif weekly_trend < -self.weekly_trend_threshold:
            strength = min(1.0, abs(weekly_trend) / 0.08)
            cut = 1.0 - (1.0 - self.weekly_bearish_cut) * strength
            return baseline * max(self.weekly_bearish_cut, cut)
        return baseline
    
    # ──── Layer 3: Sector Momentum (V7 NEW) ────
    def _apply_sector_momentum(
        self, baseline: float,
        weekly_trend: Optional[float],
        sector_trend: Optional[float]
    ) -> float:
        """
        V7: Asymmetric sector confirmation — boost only, no extra cuts.
        
        Both bullish → confirmed rally, boost baseline.
        Divergence → no adjustment (VIX/drawdown handle defense).
        """
        if sector_trend is None or np.isnan(sector_trend):
            return baseline
        if weekly_trend is None or np.isnan(weekly_trend):
            return baseline
        
        stock_bullish = weekly_trend > self.weekly_trend_threshold
        sector_bullish = sector_trend > self.weekly_trend_threshold
        
        if stock_bullish and sector_bullish:
            # Both confirm bullish — high conviction boost
            return min(MAX_POSITION, baseline * self.sector_confirm_boost)
        
        return baseline
    
    # ──── Layer 4: VIX + Term Structure ────
    def _apply_vix(
        self, baseline: float,
        vix_value: Optional[float],
        vix_term_ratio: Optional[float] = None
    ) -> float:
        if vix_value is None or np.isnan(vix_value):
            return baseline
        
        adjusted = baseline
        if vix_value > self.vix_fear_level:
            adjusted *= 0.5
        elif vix_value > self.vix_caution_level:
            factor = 1.0 - 0.2 * (vix_value - self.vix_caution_level) / (self.vix_fear_level - self.vix_caution_level)
            adjusted *= max(0.5, factor)
        elif vix_value < 15:
            adjusted = min(MAX_POSITION, adjusted * 1.1)
        
        # V7: VIX term structure — inverted curve = extra stress
        if vix_term_ratio is not None and not np.isnan(vix_term_ratio):
            if vix_term_ratio > 1.08:
                # VIX > VIX3M by >8% — inverted, near-term stress
                adjusted *= max(0.85, 1.0 - (vix_term_ratio - 1.0) * 0.3)
        
        return adjusted
    
    # ──── Layer 5: Correlation → Overlay Damping (V7 NEW) ────
    def _compute_correlation_overlay_factor(self, avg_correlation: Optional[float]) -> float:
        """
        When all stocks move together, the alpha signal is less
        reliable (systemic move, not stock-specific). Scale down
        the overlay, NOT the baseline.
        
        This avoids redundant defensive stacking with VIX/drawdown.
        """
        if avg_correlation is None or np.isnan(avg_correlation):
            return 1.0
        
        if avg_correlation > self.corr_threshold:
            # Dampen overlay — systemic move, alpha is noise
            return self.corr_overlay_dampen
        return 1.0
    
    # ──── Layer 6: Drawdown Scaling ────
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
    
    # ──── Main Position Engine ────
    def compute_position(
        self,
        action: float,
        confidence: float,
        current_volatility: float,
        regime_probs: Dict[str, float],
        vix_value: Optional[float] = None,
        vix_term_ratio: Optional[float] = None,
        weekly_trend: Optional[float] = None,
        sector_trend: Optional[float] = None,
        avg_correlation: Optional[float] = None,
    ) -> ControllerOutput:
        """6-layer position computation pipeline."""
        
        # 0. Raw baseline
        raw_baseline = self._compute_baseline(regime_probs)
        
        # 1. Regime momentum
        momentum = self._compute_regime_momentum(regime_probs)
        baseline = self._apply_regime_momentum(raw_baseline, momentum)
        
        # 2. Weekly stock trend
        baseline = self._apply_weekly_trend(baseline, weekly_trend)
        
        # 3. Sector momentum (V7)
        baseline = self._apply_sector_momentum(baseline, weekly_trend, sector_trend)
        
        # 4. VIX + term structure
        baseline = self._apply_vix(baseline, vix_value, vix_term_ratio)
        
        # 5. Drawdown scaling
        drawdown = self._compute_drawdown()
        baseline = self._apply_drawdown_scaling(baseline, drawdown)
        
        # Alpha overlay (correlation dampens overlay, not baseline)
        risk_budget = self._compute_risk_budget(drawdown)
        corr_factor = self._compute_correlation_overlay_factor(avg_correlation)
        overlay = action * confidence * OVERLAY_SCALE * risk_budget * corr_factor
        if overlay > 0:
            overlay *= 1.1
        else:
            overlay *= 0.9
        overlay = np.clip(overlay, -OVERLAY_SCALE, OVERLAY_SCALE)
        
        target = np.clip(baseline + overlay, MIN_POSITION, MAX_POSITION)
        is_trade_allowed = abs(target - self.current_position) >= self.min_trade_size
        
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
                'sector_trend': sector_trend,
                'vix_value': vix_value,
                'vix_term_ratio': vix_term_ratio,
                'avg_correlation': avg_correlation,
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
