"""
Agents package for MoA Trading Framework
"""

from .base_agent import BaseAgent, AgentSignal, SignalAggregator
from .trend_agent import TrendAgent
from .mean_reversion_agent import MeanReversionAgent
from .volatility_agent import VolatilityAgent
from .crisis_agent import CrisisAgent

__all__ = [
    'BaseAgent',
    'AgentSignal', 
    'SignalAggregator',
    'TrendAgent',
    'MeanReversionAgent',
    'VolatilityAgent',
    'CrisisAgent'
]
