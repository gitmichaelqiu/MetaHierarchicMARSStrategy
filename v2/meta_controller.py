"""
V2 Meta-Controller for MoA Trading Framework
Streamlined risk-aware position sizing.

Key V2 improvements over V1:
- Removed Kelly criterion layer (was an extra dampening multiply)
- Removed position smoothing (was 0.15 EMA, too slow for trending markets)
- Kept drawdown-based risk budget with higher floor: max(0.4, budget) vs max(0.1)
- Lower switching cost lambda: 0.005 vs 0.02 for faster repositioning
- Position = ensemble_action * confidence * risk_budget (direct, transparent)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class ControllerOutput:
    """Output from the meta-controller."""
    target_position: float
    position_change: float
    risk_budget: float
    transaction_cost_estimate: float
    is_trade_allowed: bool
    debug_info: Dict


class MetaController:
    """
    V2 Streamlined Meta-Controller.
    Position = action * confidence * risk_budget.
    No Kelly, no smoothing — fast and responsive.
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        switching_cost_lambda: float = 0.005,   # V2: lower (was 0.02)
        max_position: float = 1.0,
        min_trade_size: float = 0.02,
        drawdown_limit: float = 0.20,
        risk_budget_floor: float = 0.4,         # V2: higher floor (was 0.1)
        lookback_days: int = 20
    ):
        self.transaction_cost = transaction_cost
        self.switching_cost_lambda = switching_cost_lambda
        self.max_position = max_position
        self.min_trade_size = min_trade_size
        self.drawdown_limit = drawdown_limit
        self.risk_budget_floor = risk_budget_floor
        self.lookback_days = lookback_days
        
        # State tracking
        self.current_position = 0.0
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.position_history: deque = deque(maxlen=lookback_days)
        self.return_history: deque = deque(maxlen=lookback_days)
        
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def _calculate_risk_budget(self) -> float:
        """
        V2: Drawdown-based risk budget with higher floor.
        Still reduces risk as drawdown increases, but doesn't go below 0.4.
        """
        drawdown = self._calculate_drawdown()
        
        if drawdown <= 0:
            return 1.0
        
        if drawdown < self.drawdown_limit:
            budget = 1.0 - (drawdown / self.drawdown_limit) * 0.5
        else:
            budget = 0.5 * np.exp(-(drawdown - self.drawdown_limit) * 5)
        
        return max(self.risk_budget_floor, budget)
    
    def _apply_switching_cost_filter(
        self, 
        target_position: float, 
        signal_strength: float
    ) -> Tuple[float, bool]:
        """
        V2: Lighter switching cost filter.
        Only blocks very small or clearly unprofitable trades.
        """
        position_change = abs(target_position - self.current_position)
        
        if position_change < self.min_trade_size:
            return self.current_position, False
        
        cost = position_change * self.transaction_cost
        benefit = abs(signal_strength) * position_change
        
        net_benefit = benefit - (cost + self.switching_cost_lambda * position_change)
        
        if net_benefit < 0:
            return self.current_position, False
        
        return target_position, True
    
    def compute_position(
        self,
        ensemble_action: float,
        ensemble_confidence: float,
        current_volatility: float,
        regime_probs: Optional[Dict[str, float]] = None
    ) -> ControllerOutput:
        """
        V2: Direct position sizing.
        
        Position = sign(action) * |action| * confidence * risk_budget
        
        No Kelly, no smoothing — the ensemble already produces calibrated signals.
        """
        risk_budget = self._calculate_risk_budget()
        
        # V2: Direct sizing — ensemble action IS the position target
        # Scaled by confidence and risk budget only
        position_magnitude = abs(ensemble_action) * ensemble_confidence * risk_budget
        
        # Mild volatility scaling — only for extreme vol (>50% annualized)
        if current_volatility > 0.50:
            vol_scale = max(0.5, 1.0 - (current_volatility - 0.50) / 0.50)
            position_magnitude *= vol_scale
        
        # Apply direction
        target_position = np.sign(ensemble_action) * position_magnitude
        
        # Crisis override: reduce long exposure in crisis
        if regime_probs is not None:
            crisis_prob = regime_probs.get('Crisis', 0)
            if crisis_prob > 0.6 and target_position > 0:
                target_position *= (1 - crisis_prob * 0.4)
        
        # Clamp
        target_position = np.clip(target_position, -self.max_position, self.max_position)
        
        # Apply switching cost filter (lighter in V2)
        final_position, is_trade_allowed = self._apply_switching_cost_filter(
            target_position, 
            ensemble_action
        )
        
        position_change = final_position - self.current_position
        transaction_cost_estimate = abs(position_change) * self.transaction_cost
        
        self.position_history.append(final_position)
        
        return ControllerOutput(
            target_position=final_position,
            position_change=position_change,
            risk_budget=risk_budget,
            transaction_cost_estimate=transaction_cost_estimate,
            is_trade_allowed=is_trade_allowed,
            debug_info={
                'ensemble_action': ensemble_action,
                'ensemble_confidence': ensemble_confidence,
                'position_magnitude': position_magnitude,
                'current_drawdown': self._calculate_drawdown(),
                'current_volatility': current_volatility,
                'risk_budget': risk_budget
            }
        )
    
    def update_equity(self, daily_return: float) -> None:
        """Update equity tracking after each day."""
        self.current_equity *= (1 + daily_return)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        self.return_history.append(daily_return)
    
    def execute_trade(self, new_position: float) -> None:
        """Execute a trade (update position state)."""
        self.current_position = new_position
    
    def reset(self) -> None:
        """Reset controller state for new backtest."""
        self.current_position = 0.0
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.position_history.clear()
        self.return_history.clear()
