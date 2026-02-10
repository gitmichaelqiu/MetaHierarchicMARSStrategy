"""
V3 ExponentialMomentumAgent — Boosted signals + structural bullish bias.

Key V3 changes over V2:
- 2-of-3 bullish alignment → 0.6 signal (was 0.4)
- Structural bullish bias: if 50 EMA > 200 SMA, add +0.15 to action
- Higher base confidence for aligned signals
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
    V3 Exponential Momentum Agent — Boosted Alpha Generator.
    
    Changes from V2:
    - Stronger signals for partial alignment (0.6 vs 0.4)
    - Structural bullish bias when long-term trend is up
    """
    
    def __init__(
        self,
        ema_fast: int = 10,
        ema_mid: int = 21,
        ema_slow: int = 50,
        sma_long: int = 200,
        roc_period: int = 10
    ):
        super().__init__(
            name="ExponentialMomentumAgent",
            target_regimes=['Growth', 'Crisis']
        )
        
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.sma_long = sma_long
        self.roc_period = roc_period
    
    def _calculate_ema_cascade(self, df: pd.DataFrame) -> tuple:
        """Multi-timeframe EMA alignment signal."""
        close = df['close']
        
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_mid = close.ewm(span=self.ema_mid, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        
        price = close.iloc[-1]
        ef = ema_fast.iloc[-1]
        em = ema_mid.iloc[-1]
        es = ema_slow.iloc[-1]
        
        bullish_checks = [price > ef, ef > em, em > es]
        bearish_checks = [price < ef, ef < em, em < es]
        
        bull_count = sum(bullish_checks)
        bear_count = sum(bearish_checks)
        
        if bull_count == 3:
            spread = (ef - es) / es if es > 0 else 0
            signal = min(1.0, 0.7 + abs(spread) * 5)
            alignment = 1.0
        elif bull_count == 2:
            signal = 0.6  # V3: boosted from 0.4
            alignment = 0.75  # V3: boosted from 0.67
        elif bear_count == 3:
            spread = (es - ef) / es if es > 0 else 0
            signal = -min(1.0, 0.7 + abs(spread) * 5)
            alignment = 1.0
        elif bear_count == 2:
            signal = -0.5  # V3: boosted from -0.4
            alignment = 0.70
        else:
            signal = 0.0
            alignment = 0.0
        
        return signal, alignment
    
    def _calculate_structural_bias(self, df: pd.DataFrame) -> float:
        """
        V3 NEW: Structural bullish bias when long-term trend is up.
        If 50 EMA > 200 SMA → add +0.15 bullish tilt.
        """
        if len(df) < self.sma_long:
            return 0.0
        
        close = df['close']
        ema_50 = close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
        sma_200 = close.rolling(self.sma_long).mean().iloc[-1]
        
        if pd.isna(sma_200):
            return 0.0
        
        if ema_50 > sma_200:
            # Long-term uptrend — bullish bias
            spread = (ema_50 - sma_200) / sma_200
            return min(0.20, 0.15 + spread * 2)
        elif ema_50 < sma_200 * 0.97:
            # Long-term downtrend — slight bearish bias
            return -0.10
        
        return 0.0
    
    def _calculate_roc_signal(self, df: pd.DataFrame) -> float:
        if len(df) < self.roc_period + 1:
            return 0.0
        roc = (df['close'].iloc[-1] / df['close'].iloc[-self.roc_period] - 1) * 100
        return np.clip(np.tanh(roc / 5), -1, 1)
    
    def _calculate_efficiency_ratio(self, df: pd.DataFrame) -> float:
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
        """Generate momentum signal with boosted magnitudes and structural bias."""
        if len(df) < self.ema_slow + 5:
            return AgentSignal(action=0.0, confidence=0.0, regime_fit=0.5)
        
        # EMA cascade
        ema_signal, alignment = self._calculate_ema_cascade(df)
        
        # ROC confirmation
        roc_signal = self._calculate_roc_signal(df)
        
        # Structural bias (V3 new)
        structural_bias = self._calculate_structural_bias(df)
        
        # Efficiency ratio
        efficiency = self._calculate_efficiency_ratio(df)
        
        # Combine: EMA primary, ROC confirms
        if np.sign(ema_signal) == np.sign(roc_signal) and abs(ema_signal) > 0.3:
            action = ema_signal * 0.7 + roc_signal * 0.3
        elif abs(ema_signal) > 0.5:
            action = ema_signal * 0.8
        else:
            action = ema_signal * 0.5 + roc_signal * 0.2
        
        # V3: Add structural bias
        action = action + structural_bias
        action = np.clip(action, -1.0, 1.0)
        
        # Confidence
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
                'structural_bias': structural_bias,
                'alignment': alignment,
                'efficiency': efficiency,
            }
        )
        
        self.update_history(signal)
        return signal
