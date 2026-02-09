"""
Crisis Agent for MoA Trading Framework
Specializes in detecting and profiting from market crashes.
Acts as tail-risk protection and "Crisis Alpha" generator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base_agent import BaseAgent, AgentSignal


class CrisisAgent(BaseAgent):
    """
    Crisis Agent - The Insurance Policy
    
    Target Regime: Crisis (extreme market stress)
    
    Strategy:
    - Detects market stress indicators (vol spike, correlation breakdown)
    - Aggressive short positioning during confirmed crisis
    - Cash preservation (flat) during normal markets
    - Quick to recognize crisis onset via extreme returns
    """
    
    def __init__(
        self,
        vol_spike_threshold: float = 0.40,  # Annualized vol threshold
        extreme_return_threshold: float = -0.03,  # -3% daily return
        drawdown_threshold: float = -0.10,  # -10% drawdown
        crisis_lookback: int = 5,
        recovery_confirmation: int = 10
    ):
        """
        Initialize crisis agent.
        
        Args:
            vol_spike_threshold: Volatility above this indicates stress
            extreme_return_threshold: Daily return below this is extreme
            drawdown_threshold: Drawdown beyond this triggers defense
            crisis_lookback: Days to look back for crisis signals
            recovery_confirmation: Days of recovery before exiting crisis mode
        """
        super().__init__(
            name="CrisisAgent",
            target_regimes=['Crisis']
        )
        
        self.vol_spike_threshold = vol_spike_threshold
        self.extreme_return_threshold = extreme_return_threshold
        self.drawdown_threshold = drawdown_threshold
        self.crisis_lookback = crisis_lookback
        self.recovery_confirmation = recovery_confirmation
        
        # State tracking
        self._in_crisis_mode = False
        self._crisis_start_idx = None
    
    def _detect_volatility_spike(self, df: pd.DataFrame) -> tuple:
        """
        Detect if volatility is at crisis levels.
        Returns (is_spike: bool, severity: 0-1).
        """
        if 'hist_vol' not in df.columns:
            return False, 0.0
        
        vol = df['hist_vol'].iloc[-1]
        if pd.isna(vol):
            return False, 0.0
        
        is_spike = vol > self.vol_spike_threshold
        
        if is_spike:
            # Severity scales with how much above threshold
            severity = min(1.0, (vol - self.vol_spike_threshold) / self.vol_spike_threshold)
            return True, severity
        
        return False, 0.0
    
    def _detect_extreme_returns(self, df: pd.DataFrame) -> tuple:
        """
        Detect extreme negative returns.
        Returns (count: int, severity: 0-1).
        """
        if 'returns' not in df.columns:
            return 0, 0.0
        
        recent_returns = df['returns'].iloc[-self.crisis_lookback:]
        extreme_days = (recent_returns < self.extreme_return_threshold).sum()
        
        if extreme_days > 0:
            worst_return = recent_returns.min()
            severity = min(1.0, abs(worst_return / self.extreme_return_threshold) / 3)
            return extreme_days, severity
        
        return 0, 0.0
    
    def _calculate_drawdown(self, df: pd.DataFrame) -> float:
        """
        Calculate current drawdown from recent high.
        Returns drawdown as negative percentage.
        """
        if len(df) < 50:
            return 0.0
        
        rolling_max = df['close'].rolling(window=50, min_periods=1).max()
        drawdown = (df['close'].iloc[-1] / rolling_max.iloc[-1]) - 1
        return drawdown
    
    def _detect_trend_breakdown(self, df: pd.DataFrame) -> tuple:
        """
        Detect if price is breaking below key support levels.
        Returns (is_breakdown: bool, severity: 0-1).
        """
        if 'sma_50' not in df.columns:
            return False, 0.0
        
        close = df['close'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        
        if pd.isna(sma_50):
            return False, 0.0
        
        if close < sma_50:
            # Below 50-day MA
            distance = (sma_50 - close) / sma_50
            severity = min(1.0, distance / 0.10)  # 10% below = max severity
            
            # Check if also below 200-day if available
            if 'sma_200' in df.columns and not pd.isna(df['sma_200'].iloc[-1]):
                if close < df['sma_200'].iloc[-1]:
                    severity = min(1.0, severity + 0.3)
            
            return True, severity
        
        return False, 0.0
    
    def _calculate_crisis_score(self, df: pd.DataFrame) -> float:
        """
        Calculate composite crisis score.
        Returns value in [0, 1], higher = more crisis-like.
        """
        score = 0.0
        
        # Volatility spike
        is_vol_spike, vol_severity = self._detect_volatility_spike(df)
        if is_vol_spike:
            score += 0.3 * (1 + vol_severity)
        
        # Extreme returns
        extreme_count, return_severity = self._detect_extreme_returns(df)
        if extreme_count > 0:
            score += 0.25 * (extreme_count / self.crisis_lookback + return_severity)
        
        # Drawdown
        drawdown = self._calculate_drawdown(df)
        if drawdown < self.drawdown_threshold:
            dd_severity = min(1.0, abs(drawdown / self.drawdown_threshold) / 2)
            score += 0.25 * dd_severity
        
        # Trend breakdown
        is_breakdown, breakdown_severity = self._detect_trend_breakdown(df)
        if is_breakdown:
            score += 0.2 * breakdown_severity
        
        return min(1.0, score)
    
    def _check_recovery(self, df: pd.DataFrame) -> bool:
        """
        Check if market is recovering from crisis.
        """
        if len(df) < self.recovery_confirmation:
            return False
        
        recent = df.iloc[-self.recovery_confirmation:]
        
        # Positive returns
        positive_days = (recent['returns'] > 0).sum()
        
        # Volatility declining
        if 'hist_vol' in df.columns:
            vol_declining = recent['hist_vol'].iloc[-1] < recent['hist_vol'].iloc[0]
        else:
            vol_declining = True
        
        # Above short-term MA
        if 'sma_20' in df.columns:
            above_sma = df['close'].iloc[-1] > df['sma_20'].iloc[-1]
        else:
            above_sma = True
        
        return positive_days >= self.recovery_confirmation * 0.6 and vol_declining and above_sma
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        regime_probs: Optional[Dict[str, float]] = None
    ) -> AgentSignal:
        """
        Generate crisis protection/alpha signal.
        
        Stays flat in normal markets, goes short in crisis.
        """
        if len(df) < 50:
            return AgentSignal(action=0.0, confidence=0.0, regime_fit=0.5)
        
        # Calculate crisis score
        crisis_score = self._calculate_crisis_score(df)
        
        # Check for recovery if in crisis mode
        if self._in_crisis_mode:
            if self._check_recovery(df):
                self._in_crisis_mode = False
                self._crisis_start_idx = None
        
        # Enter crisis mode if score is high
        if crisis_score > 0.5 and not self._in_crisis_mode:
            self._in_crisis_mode = True
            self._crisis_start_idx = len(df) - 1
        
        # Generate action
        if self._in_crisis_mode or crisis_score > 0.6:
            # Crisis mode - go short
            action = -crisis_score
            confidence = 0.5 + crisis_score * 0.5
        elif crisis_score > 0.3:
            # Warning zone - reduce exposure
            action = -crisis_score * 0.5
            confidence = 0.3 + crisis_score * 0.3
        else:
            # Normal - stay flat (don't interfere with other agents)
            action = 0.0
            confidence = 0.2
        
        # Calculate regime fit
        regime_fit = self.calculate_regime_fit(regime_probs)
        
        # Get component scores for metadata
        is_vol_spike, vol_severity = self._detect_volatility_spike(df)
        extreme_count, return_severity = self._detect_extreme_returns(df)
        drawdown = self._calculate_drawdown(df)
        is_breakdown, breakdown_severity = self._detect_trend_breakdown(df)
        
        signal = AgentSignal(
            action=action,
            confidence=confidence,
            regime_fit=regime_fit,
            metadata={
                'crisis_score': crisis_score,
                'in_crisis_mode': self._in_crisis_mode,
                'vol_spike': is_vol_spike,
                'vol_severity': vol_severity,
                'extreme_return_days': extreme_count,
                'drawdown': drawdown,
                'trend_breakdown': is_breakdown,
                'hist_vol': df['hist_vol'].iloc[-1] if 'hist_vol' in df.columns else None
            }
        )
        
        self.update_history(signal)
        return signal


if __name__ == "__main__":
    # Test the crisis agent
    import sys
    sys.path.insert(0, '..')
    from data_loader import DataLoader
    from indicators import add_all_indicators
    
    loader = DataLoader()
    df = loader.fetch_ticker('AAPL', period='2y')  # Get longer period to catch potential crises
    df = add_all_indicators(df)
    
    agent = CrisisAgent()
    signal = agent.generate_signal(df)
    
    print(f"Crisis Agent Signal:")
    print(f"  Action: {signal.action:.3f}")
    print(f"  Confidence: {signal.confidence:.3f}")
    print(f"  Regime Fit: {signal.regime_fit:.3f}")
    print(f"  Metadata: {signal.metadata}")
    
    # Also test on a crisis period if we can identify one
    print(f"\n  In Crisis Mode: {agent._in_crisis_mode}")
