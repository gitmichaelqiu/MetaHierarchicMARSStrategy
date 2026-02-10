"""
Exponential Momentum Agent for MoA Trading Framework
Specializes in Hyper-Growth regimes where prices exhibit parabolic moves.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base_agent import BaseAgent, AgentSignal


class ExponentialMomentumAgent(BaseAgent):
    """
    Exponential Momentum Agent - The "Rocket" Rider
    
    Target Regimes: Growth (specifically Hyper-Growth/Parabolic)
    
    Strategy:
    - Uses ultra-fast EMAs (8/21) for trend detection.
    - Treats high RSI (>70) as a continuation signal (momentum), not reversal.
    - Uses Efficiency Ratio to validate trend quality.
    - Aggressive sizing during low-volatility strong trends.
    """
    
    def __init__(
        self,
        fast_ema: int = 8,
        slow_ema: int = 21,
        rsi_momentum_threshold: float = 60.0,
        efficiency_threshold: float = 0.1
    ):
        """
        Initialize exponential momentum agent.
        
        Args:
            fast_ema: Fast EMA period (default 8)
            slow_ema: Slow EMA period (default 21)
            rsi_momentum_threshold: RSI level to consider "strong momentum"
            efficiency_threshold: Efficiency ratio threshold for activation
        """
        super().__init__(
            name="ExponentialMomentumAgent",
            target_regimes=['Growth']
        )
        
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.rsi_momentum_threshold = rsi_momentum_threshold
        self.efficiency_threshold = efficiency_threshold
    
    def _calculate_ema_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate signal from EMA crossover.
        """
        close = df['close']
        
        # Calculate EMAs
        ema_fast = close.ewm(span=self.fast_ema, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_ema, adjust=False).mean()
        
        fast = ema_fast.iloc[-1]
        slow = ema_slow.iloc[-1]
        
        if pd.isna(fast) or pd.isna(slow):
            return 0.0
        
        # Percentage gap
        gap = (fast - slow) / slow
        
        # Normalized signal based on gap width (0 to >1)
        # 2% gap is considered strong
        signal = np.clip(gap * 50, -1, 1)
        
        # Add boost if price is above fast EMA (super strong)
        if close.iloc[-1] > fast:
            signal = min(1.0, signal * 1.2)
            
        return signal
    
    def _calculate_rsi_momentum(self, df: pd.DataFrame) -> float:
        """
        Use RSI as momentum signal.
        """
        if 'rsi' not in df.columns:
            return 0.0
            
        rsi = df['rsi'].iloc[-1]
        if pd.isna(rsi):
            return 0.0
        
        # Momentum zone: RSI 60-80 is sweet spot
        # Above 85 might be blowoff top (risk), below 50 is weak
        
        if rsi > self.rsi_momentum_threshold:
            # Strong momentum
            if rsi > 85:
                # Extreme overbought - caution but still bullish
                return 0.5
            return 1.0
        elif rsi < 40:
            # Bearish momentum
            return -0.5
        else:
            return 0.0
    
    def _calculate_efficiency(self, df: pd.DataFrame) -> float:
        """
        Calculate price efficiency (direction / volatility).
        """
        # Close-to-Close returns
        period = 20
        if len(df) < period:
            return 0.0
            
        change = (df['close'].iloc[-1] - df['close'].iloc[-period])
        abs_change = change
        
        path_length = np.sum(np.abs(df['close'].diff().tail(period)))
        
        if path_length == 0:
            return 0.0
            
        er = abs_change / path_length  # Kaufman Efficiency Ratio
        
        return er
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        regime_probs: Optional[Dict[str, float]] = None
    ) -> AgentSignal:
        """
        Generate exponential momentum signal.
        """
        if len(df) < 50:
            return AgentSignal(action=0.0, confidence=0.0, regime_fit=0.5)
        
        # Calculate heuristics
        ema_signal = self._calculate_ema_signal(df)
        rsi_signal = self._calculate_rsi_momentum(df)
        efficiency = self._calculate_efficiency(df)
        
        # Core logic: Only go long if Trend is Up AND Momentum is High
        if ema_signal > 0 and rsi_signal > 0:
            raw_action = (ema_signal * 0.6 + rsi_signal * 0.4)
            
            # Boost confidence if efficiency is high (smooth trend)
            confidence = 0.5 + (efficiency * 0.5)
            
            # Adjust for regime
            regime_fit = self.calculate_regime_fit(regime_probs)
            
            # Additional check: If in confirmed Growth regime, boost confidence
            if regime_probs and regime_probs.get('Growth', 0) > 0.6:
                confidence = min(1.0, confidence * 1.2)
                
        elif ema_signal < -0.2:
            # Weak short signal (this agent prefers long only mostly)
            raw_action = -0.3
            confidence = 0.3
            regime_fit = 0.5
        else:
            raw_action = 0.0
            confidence = 0.0
            regime_fit = 0.0
            
        # Create signal
        signal = AgentSignal(
            action=raw_action,
            confidence=confidence,
            regime_fit=regime_fit,
            metadata={
                'ema_signal': ema_signal,
                'rsi_signal': rsi_signal,
                'efficiency': efficiency
            }
        )
        
        self.update_history(signal)
        return signal


if __name__ == "__main__":
    # Test
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data_loader import DataLoader
    from indicators import add_all_indicators
    
    loader = DataLoader()
    df = loader.fetch_ticker('NVDA', period='1y')
    df = add_all_indicators(df)
    
    agent = ExponentialMomentumAgent()
    signal = agent.generate_signal(df, {'Growth': 0.8})
    
    print(f"Signal: {signal}")
