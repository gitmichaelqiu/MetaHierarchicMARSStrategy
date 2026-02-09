"""
Volatility/Breakout Agent for MoA Trading Framework
Specializes in Transition regimes where direction is unclear but volatility is building.
Positions for breakouts and profits from volatility expansion.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base_agent import BaseAgent, AgentSignal


class VolatilityAgent(BaseAgent):
    """
    Volatility/Breakout Agent - The Transition Specialist
    
    Target Regime: Transition (volatile, uncertain direction)
    
    Strategy:
    - ATR-based breakout detection
    - Volatility expansion patterns
    - Straddle-like positioning in uncertainty
    - Quick profit-taking on volatility spikes
    """
    
    def __init__(
        self,
        atr_multiplier: float = 1.5,
        vol_expansion_threshold: float = 1.3,
        lookback: int = 20,
        breakout_confirmation: int = 2
    ):
        """
        Initialize volatility agent.
        
        Args:
            atr_multiplier: ATR multiplier for breakout detection
            vol_expansion_threshold: Vol ratio above this indicates expansion
            lookback: Period for range calculation
            breakout_confirmation: Days to confirm breakout
        """
        super().__init__(
            name="VolatilityAgent",
            target_regimes=['Transition']
        )
        
        self.atr_multiplier = atr_multiplier
        self.vol_expansion_threshold = vol_expansion_threshold
        self.lookback = lookback
        self.breakout_confirmation = breakout_confirmation
    
    def _detect_range_breakout(self, df: pd.DataFrame) -> tuple:
        """
        Detect if price is breaking out of recent range.
        Returns (direction: -1/0/1, strength: 0-1).
        """
        if len(df) < self.lookback:
            return 0, 0.0
        
        close = df['close'].iloc[-1]
        recent = df.iloc[-self.lookback:-1]  # Exclude current bar
        
        high_range = recent['high'].max()
        low_range = recent['low'].min()
        range_size = high_range - low_range
        
        if range_size < 1e-6:
            return 0, 0.0
        
        # Get ATR for normalization
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
        else:
            atr = range_size / self.lookback
        
        breakout_threshold = atr * self.atr_multiplier
        
        if close > high_range:
            # Upside breakout
            strength = min(1.0, (close - high_range) / breakout_threshold)
            return 1, strength
        
        elif close < low_range:
            # Downside breakout
            strength = min(1.0, (low_range - close) / breakout_threshold)
            return -1, strength
        
        else:
            # No breakout, check position within range
            position = (close - low_range) / range_size
            # Near edges = potential breakout
            if position > 0.9:
                return 1, (position - 0.9) / 0.1 * 0.3  # Weak upside bias
            elif position < 0.1:
                return -1, (0.1 - position) / 0.1 * 0.3  # Weak downside bias
            return 0, 0.0
    
    def _detect_volatility_expansion(self, df: pd.DataFrame) -> tuple:
        """
        Detect if volatility is expanding.
        Returns (is_expanding: bool, expansion_rate: float).
        """
        if 'vol_ratio' not in df.columns:
            return False, 0.0
        
        vol_ratio = df['vol_ratio'].iloc[-1]
        if pd.isna(vol_ratio):
            return False, 0.0
        
        is_expanding = vol_ratio > self.vol_expansion_threshold
        expansion_rate = max(0, (vol_ratio - 1.0) / self.vol_expansion_threshold)
        
        return is_expanding, min(1.0, expansion_rate)
    
    def _calculate_momentum_alignment(self, df: pd.DataFrame) -> float:
        """
        Calculate if price momentum aligns with breakout direction.
        Returns value in [-1, 1].
        """
        if len(df) < 5:
            return 0.0
        
        # Recent price changes
        changes = df['close'].diff().iloc[-5:]
        
        # Count positive vs negative moves
        positive = sum(changes > 0)
        negative = sum(changes < 0)
        
        # Direction and consistency
        if positive > negative:
            consistency = (positive - negative) / 5
            return consistency
        elif negative > positive:
            consistency = (negative - positive) / 5
            return -consistency
        return 0.0
    
    def _calculate_volatility_regime_confidence(self, df: pd.DataFrame) -> float:
        """
        Calculate confidence based on volatility environment.
        """
        is_expanding, expansion_rate = self._detect_volatility_expansion(df)
        
        if is_expanding:
            # High confidence when volatility is expanding
            return 0.5 + expansion_rate * 0.5
        else:
            # Lower confidence in stable volatility
            return 0.3
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        regime_probs: Optional[Dict[str, float]] = None
    ) -> AgentSignal:
        """
        Generate volatility/breakout trading signal.
        
        Positions for breakouts, reduces exposure in uncertainty.
        """
        if len(df) < self.lookback + 5:
            return AgentSignal(action=0.0, confidence=0.0, regime_fit=0.5)
        
        # Detect breakout
        breakout_dir, breakout_strength = self._detect_range_breakout(df)
        
        # Detect volatility expansion
        is_expanding, expansion_rate = self._detect_volatility_expansion(df)
        
        # Get momentum alignment
        momentum = self._calculate_momentum_alignment(df)
        
        # Calculate action
        if breakout_dir != 0 and breakout_strength > 0.3:
            # Clear breakout - follow it
            action = breakout_dir * breakout_strength
            
            # Boost if momentum aligns
            if np.sign(momentum) == breakout_dir:
                action *= (1 + abs(momentum) * 0.3)
        
        elif is_expanding:
            # Volatility expanding but no clear direction
            # Take smaller position in momentum direction
            action = momentum * expansion_rate * 0.5
        
        else:
            # No signal
            action = 0.0
        
        # Clamp action
        action = np.clip(action, -1.0, 1.0)
        
        # Calculate confidence
        confidence = self._calculate_volatility_regime_confidence(df)
        if breakout_dir != 0:
            confidence = min(1.0, confidence + breakout_strength * 0.3)
        
        # Calculate regime fit
        regime_fit = self.calculate_regime_fit(regime_probs)
        
        signal = AgentSignal(
            action=action,
            confidence=confidence,
            regime_fit=regime_fit,
            metadata={
                'breakout_direction': breakout_dir,
                'breakout_strength': breakout_strength,
                'is_vol_expanding': is_expanding,
                'expansion_rate': expansion_rate,
                'momentum': momentum,
                'vol_ratio': df['vol_ratio'].iloc[-1] if 'vol_ratio' in df.columns else None
            }
        )
        
        self.update_history(signal)
        return signal


if __name__ == "__main__":
    # Test the volatility agent
    import sys
    sys.path.insert(0, '..')
    from data_loader import DataLoader
    from indicators import add_all_indicators
    
    loader = DataLoader()
    df = loader.fetch_ticker('AAPL', period='6mo')
    df = add_all_indicators(df)
    
    agent = VolatilityAgent()
    signal = agent.generate_signal(df)
    
    print(f"Volatility Agent Signal:")
    print(f"  Action: {signal.action:.3f}")
    print(f"  Confidence: {signal.confidence:.3f}")
    print(f"  Regime Fit: {signal.regime_fit:.3f}")
    print(f"  Metadata: {signal.metadata}")
