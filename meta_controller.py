"""
Meta-Controller Module for MoA Trading Framework
Implements risk-aware capital allocation with transaction cost penalties.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class ControllerOutput:
    """
    Output from the meta-controller.
    
    Attributes:
        target_position: Final position size in [-1, 1]
        position_change: Change from previous position
        risk_budget: Current risk budget multiplier
        transaction_cost_estimate: Estimated cost of this trade
        is_trade_allowed: Whether trading is recommended
        debug_info: Additional debugging information
    """
    target_position: float
    position_change: float
    risk_budget: float
    transaction_cost_estimate: float
    is_trade_allowed: bool
    debug_info: Dict


class MetaController:
    """
    Meta-Adaptive Controller for risk-aware position sizing.
    
    Features:
    - Drawdown-based risk budget
    - Transaction cost penalties
    - Kelly criterion-inspired sizing
    - Smooth position transitions
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1%
        switching_cost_lambda: float = 0.02,  # Reduced to allow more trades
        max_position: float = 1.0,
        min_trade_size: float = 0.03,  # Lower threshold for trades
        drawdown_limit: float = 0.20,  # 20% max drawdown (more tolerance)
        kelly_fraction: float = 1.0,  # Full Kelly for more aggressive sizing
        smoothing_factor: float = 0.15,  # Less smoothing for faster response
        min_position_floor: float = 0.3,  # Minimum position for strong signals
        lookback_days: int = 20
    ):
        """
        Initialize meta-controller.
        
        Args:
            transaction_cost: Cost per trade as fraction of trade value
            switching_cost_lambda: Penalty for position changes
            max_position: Maximum absolute position size
            min_trade_size: Minimum position change to execute
            drawdown_limit: Maximum acceptable drawdown
            kelly_fraction: Fraction of Kelly criterion to use
            smoothing_factor: Position smoothing (0 = no smooth, 1 = full smooth)
            lookback_days: Days of history for metrics
        """
        self.transaction_cost = transaction_cost
        self.switching_cost_lambda = switching_cost_lambda
        self.max_position = max_position
        self.min_trade_size = min_trade_size
        self.drawdown_limit = drawdown_limit
        self.kelly_fraction = kelly_fraction
        self.smoothing_factor = smoothing_factor
        self.min_position_floor = min_position_floor
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
        Calculate risk budget based on drawdown.
        
        Reduces risk as drawdown increases.
        """
        drawdown = self._calculate_drawdown()
        
        if drawdown <= 0:
            return 1.0  # Full risk budget
        
        # Linear risk reduction
        if drawdown < self.drawdown_limit:
            budget = 1.0 - (drawdown / self.drawdown_limit) * 0.5
        else:
            # Beyond limit: aggressive reduction
            budget = 0.5 * np.exp(-(drawdown - self.drawdown_limit) * 10)
        
        return max(0.1, budget)
    
    def _calculate_kelly_size(
        self, 
        signal_strength: float, 
        confidence: float,
        expected_volatility: float
    ) -> float:
        """
        Calculate position size using Kelly criterion variant.
        
        Kelly = (W*p - L*q) / (W*L)
        Approximated as: signal_strength * confidence / volatility_normalized
        """
        if expected_volatility <= 0:
            expected_volatility = 0.20  # Default 20% vol
        
        # Volatility scaling: reduce position in high vol (but less aggressive)
        vol_scale = min(1.0, 0.30 / expected_volatility)  # More tolerant of volatility
        
        # Kelly-inspired sizing with boost for strong signals
        kelly = abs(signal_strength) * confidence * vol_scale
        
        # Apply fraction
        position_size = kelly * self.kelly_fraction
        
        # Enforce minimum position for strong, confident signals
        if abs(signal_strength) > 0.5 and confidence > 0.4:
            position_size = max(position_size, self.min_position_floor * confidence)
        
        return min(1.0, position_size)
    
    def _apply_switching_cost_filter(
        self, 
        target_position: float, 
        signal_strength: float
    ) -> Tuple[float, bool]:
        """
        Apply switching cost filter to reduce unnecessary trades.
        
        Only trade if expected benefit exceeds cost.
        
        Returns:
            (adjusted_position, is_trade_worthwhile)
        """
        position_change = abs(target_position - self.current_position)
        
        if position_change < self.min_trade_size:
            return self.current_position, False
        
        # Estimate transaction cost
        cost = position_change * self.transaction_cost
        
        # Estimate benefit (signal strength as proxy)
        benefit = abs(signal_strength) * position_change
        
        # Apply switching cost penalty
        net_benefit = benefit - (cost + self.switching_cost_lambda * position_change)
        
        if net_benefit < 0:
            # Not worth trading
            return self.current_position, False
        
        return target_position, True
    
    def _smooth_position(self, target_position: float) -> float:
        """
        Smooth position changes to reduce whipsaw.
        """
        if self.smoothing_factor <= 0:
            return target_position
        
        # Exponential smoothing
        smoothed = (
            self.smoothing_factor * self.current_position + 
            (1 - self.smoothing_factor) * target_position
        )
        
        return smoothed
    
    def compute_position(
        self,
        ensemble_action: float,
        ensemble_confidence: float,
        current_volatility: float,
        regime_probs: Optional[Dict[str, float]] = None
    ) -> ControllerOutput:
        """
        Compute the final position size.
        
        Args:
            ensemble_action: Combined action from ensemble [-1, 1]
            ensemble_confidence: Overall confidence [0, 1]
            current_volatility: Current annualized volatility
            regime_probs: Optional regime probabilities for crisis override
            
        Returns:
            ControllerOutput with target position and metadata
        """
        # Calculate risk budget
        risk_budget = self._calculate_risk_budget()
        
        # Calculate Kelly-inspired position size
        position_magnitude = self._calculate_kelly_size(
            ensemble_action, 
            ensemble_confidence,
            current_volatility
        )
        
        # Apply risk budget
        position_magnitude *= risk_budget
        
        # Apply direction
        target_position = np.sign(ensemble_action) * position_magnitude
        
        # Crisis override: reduce exposure in crisis regime
        if regime_probs is not None:
            crisis_prob = regime_probs.get('Crisis', 0)
            if crisis_prob > 0.5:
                # High crisis probability - reduce long exposure
                if target_position > 0:
                    target_position *= (1 - crisis_prob * 0.5)
        
        # Apply smoothing
        smoothed_position = self._smooth_position(target_position)
        
        # Clamp to max position
        smoothed_position = np.clip(smoothed_position, -self.max_position, self.max_position)
        
        # Apply switching cost filter
        final_position, is_trade_allowed = self._apply_switching_cost_filter(
            smoothed_position, 
            ensemble_action
        )
        
        # Calculate position change and cost
        position_change = final_position - self.current_position
        transaction_cost_estimate = abs(position_change) * self.transaction_cost
        
        # Update internal state
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
                'smoothed_position': smoothed_position,
                'current_drawdown': self._calculate_drawdown(),
                'current_volatility': current_volatility
            }
        )
    
    def update_equity(self, daily_return: float) -> None:
        """
        Update equity tracking after each day.
        
        Args:
            daily_return: Portfolio return for the day
        """
        self.current_equity *= (1 + daily_return)
        self.peak_equity = max(self.peak_equity, self.current_equity)
        self.return_history.append(daily_return)
    
    def execute_trade(self, new_position: float) -> None:
        """
        Execute a trade (update position state).
        
        Args:
            new_position: New position after trade
        """
        self.current_position = new_position
    
    def reset(self) -> None:
        """Reset controller state for new backtest."""
        self.current_position = 0.0
        self.peak_equity = 1.0
        self.current_equity = 1.0
        self.position_history.clear()
        self.return_history.clear()


if __name__ == "__main__":
    # Test the meta-controller
    controller = MetaController()
    
    # Simulate a series of decisions
    scenarios = [
        (0.8, 0.9, 0.20, None),     # Strong bullish signal, low vol
        (0.6, 0.7, 0.35, None),     # Moderate bullish, high vol
        (-0.5, 0.6, 0.25, {'Crisis': 0.6}),  # Bearish with crisis
        (0.2, 0.5, 0.20, None),     # Weak signal
    ]
    
    print("Meta-Controller Tests:")
    print("=" * 60)
    
    for i, (action, conf, vol, regime) in enumerate(scenarios):
        output = controller.compute_position(action, conf, vol, regime)
        
        print(f"\nScenario {i+1}:")
        print(f"  Input: action={action:.2f}, conf={conf:.2f}, vol={vol:.2f}")
        print(f"  Target Position: {output.target_position:.3f}")
        print(f"  Position Change: {output.position_change:.3f}")
        print(f"  Risk Budget: {output.risk_budget:.3f}")
        print(f"  Trade Allowed: {output.is_trade_allowed}")
        
        # Execute trade to update state
        if output.is_trade_allowed:
            controller.execute_trade(output.target_position)
            # Simulate some return
            controller.update_equity(0.01 * np.sign(action))
