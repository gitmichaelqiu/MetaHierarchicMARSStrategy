"""
Mean Reversion Agent for MoA Trading Framework
Specializes in Stagnation regimes where prices oscillate around a mean.
Acts as a liquidity provider, betting on price convergence.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base_agent import BaseAgent, AgentSignal


class MeanReversionAgent(BaseAgent):
    """
    Mean Reversion Agent - The Liquidity Provider
    
    Target Regime: Stagnation (sideways markets)
    
    Strategy:
    - Bollinger Band mean-reversion
    - RSI overbought/oversold signals
    - Z-score deviation from rolling mean
    - Tight stop-loss awareness to protect against trend breakouts
    """
    
    def __init__(
        self,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        bb_overbought: float = 0.95,
        bb_oversold: float = 0.05,
        z_score_threshold: float = 2.0,
        rsi_weight: float = 0.4,
        bb_weight: float = 0.35,
        zscore_weight: float = 0.25
    ):
        """
        Initialize mean reversion agent.
        
        Args:
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            bb_overbought: Bollinger %B above this is overbought
            bb_oversold: Bollinger %B below this is oversold
            z_score_threshold: Z-score magnitude for extreme readings
            rsi_weight: Weight for RSI signal
            bb_weight: Weight for Bollinger Band signal
            zscore_weight: Weight for Z-score signal
        """
        super().__init__(
            name="MeanReversionAgent",
            target_regimes=['Stagnation']
        )
        
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_overbought = bb_overbought
        self.bb_oversold = bb_oversold
        self.z_score_threshold = z_score_threshold
        self.rsi_weight = rsi_weight
        self.bb_weight = bb_weight
        self.zscore_weight = zscore_weight
    
    def _calculate_rsi_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate mean reversion signal from RSI.
        Returns value in [-1, 1].
        """
        if 'rsi' not in df.columns:
            return 0.0
        
        rsi = df['rsi'].iloc[-1]
        if pd.isna(rsi):
            return 0.0
        
        if rsi >= self.rsi_overbought:
            # Overbought - sell signal (short)
            # Scale based on how far above threshold
            intensity = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return -min(1.0, intensity)
        
        elif rsi <= self.rsi_oversold:
            # Oversold - buy signal (long)
            intensity = (self.rsi_oversold - rsi) / self.rsi_oversold
            return min(1.0, intensity)
        
        else:
            # Neutral zone - weak signal based on distance from 50
            deviation = (50 - rsi) / 50 * 0.3  # Scale down neutral signals
            return deviation
    
    def _calculate_bb_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate mean reversion signal from Bollinger %B.
        Returns value in [-1, 1].
        """
        if 'bb_pct_b' not in df.columns:
            return 0.0
        
        pct_b = df['bb_pct_b'].iloc[-1]
        if pd.isna(pct_b):
            return 0.0
        
        if pct_b >= self.bb_overbought:
            # Above upper band - sell signal
            intensity = (pct_b - self.bb_overbought) / (1 - self.bb_overbought)
            return -min(1.0, intensity)
        
        elif pct_b <= self.bb_oversold:
            # Below lower band - buy signal
            intensity = (self.bb_oversold - pct_b) / self.bb_oversold
            return min(1.0, intensity)
        
        else:
            # Within bands - signal based on distance from middle
            deviation = (0.5 - pct_b) * 0.5  # Scale down
            return deviation
    
    def _calculate_zscore_signal(self, df: pd.DataFrame) -> float:
        """
        Calculate mean reversion signal from Z-score.
        Returns value in [-1, 1].
        """
        if 'z_score' not in df.columns:
            return 0.0
        
        z = df['z_score'].iloc[-1]
        if pd.isna(z):
            return 0.0
        
        if abs(z) >= self.z_score_threshold:
            # Extreme deviation - strong mean reversion signal
            return -np.sign(z) * min(1.0, abs(z) / self.z_score_threshold * 0.5)
        else:
            # Moderate deviation
            return -z / self.z_score_threshold * 0.3
    
    def _calculate_volatility_regime_fit(self, df: pd.DataFrame) -> float:
        """
        Mean reversion works best in low volatility environments.
        Returns adjustment factor in [0, 1].
        """
        if 'hist_vol' in df.columns:
            vol = df['hist_vol'].iloc[-1]
            if pd.isna(vol):
                return 0.5
            # Low vol (< 20%) is ideal, high vol (> 40%) is bad
            if vol < 0.20:
                return 1.0
            elif vol > 0.40:
                return 0.2
            else:
                return 1.0 - (vol - 0.20) / 0.20 * 0.8
        
        return 0.5
    
    def _detect_breakout_risk(self, df: pd.DataFrame) -> float:
        """
        Detect if a breakout might be forming (danger for mean reversion).
        Returns risk score in [0, 1].
        """
        risk = 0.0
        
        # ADX trending up = breakout risk
        if 'adx' in df.columns and len(df) > 5:
            adx_now = df['adx'].iloc[-1]
            adx_prev = df['adx'].iloc[-5]
            if not pd.isna(adx_now) and not pd.isna(adx_prev):
                if adx_now > adx_prev and adx_now > 25:
                    risk += 0.3
        
        # Volume spike = potential breakout
        if 'volume' in df.columns and len(df) > 20:
            vol_20d = df['volume'].iloc[-20:].mean()
            vol_now = df['volume'].iloc[-1]
            if vol_now > vol_20d * 1.5:
                risk += 0.3
        
        # Volatility expanding
        if 'vol_ratio' in df.columns:
            vol_ratio = df['vol_ratio'].iloc[-1]
            if not pd.isna(vol_ratio) and vol_ratio > 1.3:
                risk += 0.2
        
        return min(1.0, risk)
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        regime_probs: Optional[Dict[str, float]] = None
    ) -> AgentSignal:
        """
        Generate mean reversion trading signal.
        
        Buy when oversold, sell when overbought.
        Reduce signal when breakout risk is high.
        """
        if len(df) < 20:
            return AgentSignal(action=0.0, confidence=0.0, regime_fit=0.5)
        
        # Calculate individual signals
        rsi_signal = self._calculate_rsi_signal(df)
        bb_signal = self._calculate_bb_signal(df)
        zscore_signal = self._calculate_zscore_signal(df)
        
        # Weighted combination
        raw_action = (
            rsi_signal * self.rsi_weight +
            bb_signal * self.bb_weight +
            zscore_signal * self.zscore_weight
        )
        
        # Adjust for volatility regime
        vol_fit = self._calculate_volatility_regime_fit(df)
        
        # Reduce action if breakout risk is high
        breakout_risk = self._detect_breakout_risk(df)
        action_multiplier = 1.0 - breakout_risk * 0.7
        
        adjusted_action = raw_action * action_multiplier
        
        # Calculate regime fit
        regime_fit = self.calculate_regime_fit(regime_probs)
        
        # Confidence based on signal agreement and volatility fit
        signals = [rsi_signal, bb_signal, zscore_signal]
        sign_agreement = sum(1 for s in signals if np.sign(s) == np.sign(raw_action)) / len(signals)
        confidence = (sign_agreement * 0.5 + vol_fit * 0.3 + (1 - breakout_risk) * 0.2)
        
        signal = AgentSignal(
            action=adjusted_action,
            confidence=confidence,
            regime_fit=regime_fit * vol_fit,  # Reduce fit in high vol
            metadata={
                'rsi_signal': rsi_signal,
                'bb_signal': bb_signal,
                'zscore_signal': zscore_signal,
                'breakout_risk': breakout_risk,
                'vol_fit': vol_fit,
                'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else None,
                'bb_pct_b': df['bb_pct_b'].iloc[-1] if 'bb_pct_b' in df.columns else None
            }
        )
        
        self.update_history(signal)
        return signal


if __name__ == "__main__":
    # Test the mean reversion agent
    import sys
    sys.path.insert(0, '..')
    from data_loader import DataLoader
    from indicators import add_all_indicators
    
    loader = DataLoader()
    df = loader.fetch_ticker('AAPL', period='6mo')
    df = add_all_indicators(df)
    
    agent = MeanReversionAgent()
    signal = agent.generate_signal(df)
    
    print(f"Mean Reversion Agent Signal:")
    print(f"  Action: {signal.action:.3f}")
    print(f"  Confidence: {signal.confidence:.3f}")
    print(f"  Regime Fit: {signal.regime_fit:.3f}")
    print(f"  Metadata: {signal.metadata}")
