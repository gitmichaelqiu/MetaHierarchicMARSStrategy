"""
Trend Following Agent for MoA Trading Framework
Specializes in Growth and Crisis regimes where prices exhibit strong trends.
Implements "Crisis Alpha" - profits from both bull trends and crash trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base_agent import BaseAgent, AgentSignal


class TrendAgent(BaseAgent):
    """
    Trend Following Agent - The Convexity Engine
    
    Target Regimes: Growth (bull trends), Crisis (crash trends)
    
    Strategy:
    - Uses MACD crossovers for trend direction
    - ADX for trend strength filtering
    - Dual moving average for confirmation
    - Automatically flips to short during strong downtrends
    """
    
    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 50,
        adx_threshold: float = 20.0,
        macd_weight: float = 0.5,
        ma_weight: float = 0.3,
        momentum_weight: float = 0.2
    ):
        """
        Initialize trend agent.
        
        Args:
            fast_ma: Fast moving average period
            slow_ma: Slow moving average period  
            adx_threshold: Minimum ADX for trend confirmation (> this = trending)
            macd_weight: Weight for MACD signal
            ma_weight: Weight for MA crossover signal
            momentum_weight: Weight for momentum signal
        """
        super().__init__(
            name="TrendAgent",
            target_regimes=['Growth', 'Crisis']
        )
        
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.adx_threshold = adx_threshold
        self.macd_weight = macd_weight
        self.ma_weight = ma_weight
        self.momentum_weight = momentum_weight
    
    def _calculate_ma_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate signal from moving average crossover.
        Returns value in [-1, 1].
        """
        close = df['close'].iloc[-1]
        
        # Calculate MAs if not present
        if 'sma_20' in df.columns:
            fast = df['sma_20'].iloc[-1]
        else:
            fast = df['close'].rolling(self.fast_ma).mean().iloc[-1]
            
        if 'sma_50' in df.columns:
            slow = df['sma_50'].iloc[-1]
        else:
            slow = df['close'].rolling(self.slow_ma).mean().iloc[-1]
        
        if pd.isna(fast) or pd.isna(slow):
            return 0.0
        
        # Price position relative to MAs
        above_fast = 1 if close > fast else -1
        above_slow = 1 if close > slow else -1
        fast_above_slow = 1 if fast > slow else -1
        
        # Combine signals
        signal = (above_fast * 0.3 + above_slow * 0.3 + fast_above_slow * 0.4)
        return np.clip(signal, -1, 1)
    
    def _calculate_macd_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate signal from MACD.
        Returns value in [-1, 1].
        """
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return 0.0
        
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else macd - macd_signal
        
        if pd.isna(macd) or pd.isna(macd_signal):
            return 0.0
        
        # Histogram direction and magnitude
        hist_normalized = np.tanh(macd_hist * 50)  # Normalize to [-1, 1]
        
        # MACD above/below signal line
        cross_signal = 1 if macd > macd_signal else -1
        
        # MACD above/below zero
        zero_signal = 1 if macd > 0 else -1
        
        signal = hist_normalized * 0.5 + cross_signal * 0.3 + zero_signal * 0.2
        return np.clip(signal, -1, 1)
    
    def _calculate_momentum_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate signal from price momentum.
        Returns value in [-1, 1].
        """
        if 'roc' in df.columns:
            roc = df['roc'].iloc[-1]
            if pd.isna(roc):
                return 0.0
            # ROC is typically in percentage terms
            return np.clip(np.tanh(roc / 10), -1, 1)
        
        # Fallback: calculate simple momentum
        if len(df) < 10:
            return 0.0
        
        returns_10d = (df['close'].iloc[-1] / df['close'].iloc[-10]) - 1
        return np.clip(np.tanh(returns_10d * 10), -1, 1)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength for confidence scaling.
        Returns value in [0, 1].
        """
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]
            if pd.isna(adx):
                return 0.5
            # ADX typically 0-60, normalize to [0, 1]
            strength = min(adx / 50, 1.0)
            return strength
        
        # Fallback: use volatility-adjusted trend
        if 'trend_strength' in df.columns:
            return df['trend_strength'].iloc[-1] / 100
        
        return 0.5
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        regime_probs: Optional[Dict[str, float]] = None
    ) -> AgentSignal:
        """
        Generate trend-following trading signal.
        
        Long in uptrends, short in downtrends.
        Signal strength based on trend conviction.
        """
        if len(df) < self.slow_ma:
            return AgentSignal(action=0.0, confidence=0.0, regime_fit=0.5)
        
        # Calculate individual signals
        ma_signal = self._calculate_ma_signal(df)
        macd_signal = self._calculate_macd_signal(df)
        momentum_signal = self._calculate_momentum_signal(df)
        
        # Weighted combination
        raw_action = (
            ma_signal * self.ma_weight +
            macd_signal * self.macd_weight +
            momentum_signal * self.momentum_weight
        )
        
        # Trend strength affects confidence
        trend_strength = self._calculate_trend_strength(df)
        
        # ADX filtering: reduce action if trend is weak
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]
            if not pd.isna(adx) and adx < self.adx_threshold:
                # Weak trend - reduce action magnitude
                raw_action *= (adx / self.adx_threshold)
        
        # Calculate regime fit
        regime_fit = self.calculate_regime_fit(regime_probs)
        
        # Confidence based on signal agreement and trend strength
        signal_agreement = 1.0 - abs(
            (abs(ma_signal) + abs(macd_signal) + abs(momentum_signal)) / 3 -
            abs(raw_action)
        )
        confidence = (trend_strength * 0.6 + signal_agreement * 0.4)
        
        # Create signal
        signal = AgentSignal(
            action=raw_action,
            confidence=confidence,
            regime_fit=regime_fit,
            metadata={
                'ma_signal': ma_signal,
                'macd_signal': macd_signal,
                'momentum_signal': momentum_signal,
                'trend_strength': trend_strength,
                'adx': df['adx'].iloc[-1] if 'adx' in df.columns else None
            }
        )
        
        self.update_history(signal)
        return signal


if __name__ == "__main__":
    # Test the trend agent
    import sys
    sys.path.insert(0, '..')
    from data_loader import DataLoader
    from indicators import add_all_indicators
    
    loader = DataLoader()
    df = loader.fetch_ticker('AAPL', period='6mo')
    df = add_all_indicators(df)
    
    agent = TrendAgent()
    signal = agent.generate_signal(df)
    
    print(f"Trend Agent Signal:")
    print(f"  Action: {signal.action:.3f}")
    print(f"  Confidence: {signal.confidence:.3f}")
    print(f"  Regime Fit: {signal.regime_fit:.3f}")
    print(f"  Metadata: {signal.metadata}")
