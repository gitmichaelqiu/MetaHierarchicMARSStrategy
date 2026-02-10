"""
Exponential Momentum Agent for MoA Trading Framework V2
Primary alpha generator during Growth regimes.

Uses multi-timeframe EMA cascade for strong directional signals.
Unlike the TrendAgent (which uses lagging SMA/MACD), this agent uses
faster exponential averages and produces higher-magnitude signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_V1_DIR = os.path.join(_PROJECT_ROOT, 'v1')
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)

from agents.base_agent import BaseAgent, AgentSignal


class ExponentialMomentumAgent(BaseAgent):
    """
    Exponential Momentum Agent - The Alpha Generator
    
    Target Regime: Growth (strong uptrends), Crisis (strong downtrends)
    
    Strategy:
    - Multi-timeframe EMA cascade (10/21/50)
    - All-timeframe alignment produces strong signals (0.7-1.0)
    - Efficiency ratio for trend quality filtering
    - ROC momentum confirmation
    """
    
    def __init__(
        self,
        ema_fast: int = 10,
        ema_mid: int = 21,
        ema_slow: int = 50,
        roc_period: int = 10
    ):
        super().__init__(
            name="ExponentialMomentumAgent",
            target_regimes=['Growth', 'Crisis']
        )
        
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.roc_period = roc_period
    
    def _calculate_ema_cascade(self, df: pd.DataFrame) -> tuple:
        """
        Calculate multi-timeframe EMA alignment signal.
        
        Returns:
            (signal: float [-1, 1], alignment_score: float [0, 1])
        """
        close = df['close']
        
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_mid = close.ewm(span=self.ema_mid, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        
        price = close.iloc[-1]
        ef = ema_fast.iloc[-1]
        em = ema_mid.iloc[-1]
        es = ema_slow.iloc[-1]
        
        # Check alignment: price > fast > mid > slow (bullish) or reverse
        bullish_checks = [
            price > ef,
            ef > em,
            em > es,
        ]
        bearish_checks = [
            price < ef,
            ef < em,
            em < es,
        ]
        
        bull_count = sum(bullish_checks)
        bear_count = sum(bearish_checks)
        
        if bull_count == 3:
            # Perfect bullish alignment
            # Magnitude based on spread between fast and slow
            spread = (ef - es) / es if es > 0 else 0
            signal = min(1.0, 0.7 + abs(spread) * 5)
            alignment = 1.0
        elif bull_count == 2:
            signal = 0.4
            alignment = 0.67
        elif bear_count == 3:
            spread = (es - ef) / es if es > 0 else 0
            signal = -min(1.0, 0.7 + abs(spread) * 5)
            alignment = 1.0
        elif bear_count == 2:
            signal = -0.4
            alignment = 0.67
        else:
            signal = 0.0
            alignment = 0.0
        
        return signal, alignment
    
    def _calculate_roc_signal(self, df: pd.DataFrame) -> float:
        """Rate of change momentum confirmation."""
        if len(df) < self.roc_period + 1:
            return 0.0
        
        roc = (df['close'].iloc[-1] / df['close'].iloc[-self.roc_period] - 1) * 100
        return np.clip(np.tanh(roc / 5), -1, 1)
    
    def _calculate_efficiency_ratio(self, df: pd.DataFrame) -> float:
        """
        Kaufman Efficiency Ratio.
        High = trending cleanly, Low = choppy/noisy.
        """
        if len(df) < self.ema_slow:
            return 0.5
        
        period = 20
        direction = abs(df['close'].iloc[-1] - df['close'].iloc[-period])
        volatility = df['close'].diff().abs().iloc[-period:].sum()
        
        if volatility < 1e-8:
            return 0.5
        
        return min(1.0, direction / volatility)
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        regime_probs: Optional[Dict[str, float]] = None
    ) -> AgentSignal:
        """
        Generate momentum signal.
        
        Strong signals when all EMA timeframes align.
        """
        if len(df) < self.ema_slow + 5:
            return AgentSignal(action=0.0, confidence=0.0, regime_fit=0.5)
        
        # EMA cascade
        ema_signal, alignment = self._calculate_ema_cascade(df)
        
        # ROC confirmation
        roc_signal = self._calculate_roc_signal(df)
        
        # Efficiency ratio as quality filter
        efficiency = self._calculate_efficiency_ratio(df)
        
        # Combine: EMA is primary, ROC confirms
        if np.sign(ema_signal) == np.sign(roc_signal) and abs(ema_signal) > 0.3:
            # Confirmed momentum — boost signal
            action = ema_signal * 0.7 + roc_signal * 0.3
        elif abs(ema_signal) > 0.5:
            # Strong EMA but weak ROC — use EMA with slight reduction
            action = ema_signal * 0.8
        else:
            action = ema_signal * 0.5 + roc_signal * 0.2
        
        action = np.clip(action, -1.0, 1.0)
        
        # Confidence: alignment + efficiency  
        confidence = alignment * 0.6 + efficiency * 0.4
        
        # Regime fit
        regime_fit = self.calculate_regime_fit(regime_probs)
        
        signal = AgentSignal(
            action=action,
            confidence=confidence,
            regime_fit=regime_fit,
            metadata={
                'ema_signal': ema_signal,
                'roc_signal': roc_signal,
                'alignment': alignment,
                'efficiency': efficiency,
            }
        )
        
        self.update_history(signal)
        return signal
