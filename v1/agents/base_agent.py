"""
Base Agent Module for MoA Trading Framework
Defines the abstract interface that all trading agents must implement.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AgentSignal:
    """
    Trading signal from an agent.
    
    Attributes:
        action: Trading action in range [-1, 1]
                -1 = full short, 0 = flat, +1 = full long
        confidence: Agent's confidence in the signal (0 to 1)
        regime_fit: How well current market fits agent's target regime (0 to 1)
        metadata: Additional debug/analysis information
    """
    action: float
    confidence: float
    regime_fit: float
    metadata: Dict = None
    
    def __post_init__(self):
        # Clamp values to valid ranges
        self.action = max(-1.0, min(1.0, self.action))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.regime_fit = max(0.0, min(1.0, self.regime_fit))
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def weighted_action(self) -> float:
        """Action weighted by confidence and regime fit."""
        return self.action * self.confidence * self.regime_fit


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Each agent specializes in specific market regimes and generates
    trading signals accordingly.
    """
    
    def __init__(self, name: str, target_regimes: List[str]):
        """
        Initialize base agent.
        
        Args:
            name: Human-readable agent name
            target_regimes: List of regimes this agent is designed for
                           ['Growth', 'Stagnation', 'Transition', 'Crisis']
        """
        self.name = name
        self.target_regimes = target_regimes
        self.last_signal: Optional[AgentSignal] = None
        self.signal_history: List[AgentSignal] = []
    
    @abstractmethod
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        regime_probs: Optional[Dict[str, float]] = None
    ) -> AgentSignal:
        """
        Generate a trading signal based on market data.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            regime_probs: Current regime probability distribution
            
        Returns:
            AgentSignal with action, confidence, and regime_fit
        """
        pass
    
    def calculate_regime_fit(self, regime_probs: Optional[Dict[str, float]]) -> float:
        """
        Calculate how well the current market fits this agent's expertise.
        
        Uses soft assignment from regime probabilities.
        """
        if regime_probs is None:
            return 0.5  # Neutral if no regime info
        
        fit = 0.0
        for regime in self.target_regimes:
            fit += regime_probs.get(regime, 0.0)
        
        return min(1.0, fit)
    
    def update_history(self, signal: AgentSignal) -> None:
        """Track signal history for analysis."""
        self.last_signal = signal
        self.signal_history.append(signal)
        
        # Keep only recent history
        if len(self.signal_history) > 252:  # ~1 year of daily data
            self.signal_history = self.signal_history[-252:]
    
    def get_average_confidence(self, window: int = 20) -> float:
        """Get average confidence over recent signals."""
        if not self.signal_history:
            return 0.5
        
        recent = self.signal_history[-window:]
        return np.mean([s.confidence for s in recent])
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', targets={self.target_regimes})"


class SignalAggregator:
    """
    Utility class to aggregate signals from multiple agents.
    """
    
    @staticmethod
    def soft_vote(signals: List[Tuple[BaseAgent, AgentSignal]], weights: Dict[str, float] = None) -> float:
        """
        Soft voting aggregation of agent signals.
        
        Args:
            signals: List of (agent, signal) tuples
            weights: Optional override weights by agent name
            
        Returns:
            Aggregated action in [-1, 1]
        """
        if not signals:
            return 0.0
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for agent, signal in signals:
            # Get weight (from override or from signal quality)
            if weights and agent.name in weights:
                w = weights[agent.name]
            else:
                w = signal.confidence * signal.regime_fit
            
            weighted_sum += signal.action * w
            weight_total += w
        
        if weight_total < 1e-6:
            return 0.0
        
        return weighted_sum / weight_total
    
    @staticmethod
    def detect_conflict(signals: List[Tuple[BaseAgent, AgentSignal]], threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Detect if agents are in significant conflict.
        
        Returns:
            (is_conflicting, conflict_score)
            conflict_score in [0, 1], higher = more conflict
        """
        if len(signals) < 2:
            return False, 0.0
        
        actions = [s.action for _, s in signals]
        
        # Check for sign disagreement
        has_long = any(a > threshold for a in actions)
        has_short = any(a < -threshold for a in actions)
        
        if has_long and has_short:
            # Conflict exists - calculate severity
            max_long = max(a for a in actions if a > 0) if has_long else 0
            max_short = min(a for a in actions if a < 0) if has_short else 0
            conflict_score = (max_long - max_short) / 2  # Normalize to [0, 1]
            return True, min(1.0, conflict_score)
        
        return False, 0.0
