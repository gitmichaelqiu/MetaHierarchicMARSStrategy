"""
V4 MoA Soft Ensemble — Simplified for overlay architecture.

In V4, the ensemble output is purely the alpha signal (direction + magnitude).
The baseline position is handled by the meta-controller.
Agreement floor removed (baseline handles minimum position).
Momentum memory kept from V3.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V1_DIR = os.path.join(_PROJECT_ROOT, 'v1')
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)

from agents.base_agent import AgentSignal


@dataclass
class EnsembleOutput:
    """Output from the ensemble."""
    final_action: float       # Alpha signal: -1 to +1
    confidence: float
    is_conflicting: bool
    conflict_penalty: float
    agent_contributions: Dict[str, float]


class MoASoftEnsemble:
    """
    V4 Ensemble — outputs alpha signal for overlay.
    
    Simplified: no agreement floor (baseline handles that).
    Kept: momentum memory, conflict detection.
    """
    
    def __init__(
        self,
        conflict_threshold: float = 0.8,
        max_conflict_penalty: float = 0.4,
        momentum_memory_window: int = 10,
        momentum_carry_factor: float = 0.4,
    ):
        self.conflict_threshold = conflict_threshold
        self.max_conflict_penalty = max_conflict_penalty
        self.momentum_memory_window = momentum_memory_window
        self.momentum_carry_factor = momentum_carry_factor
        
        self._action_history: deque = deque(maxlen=momentum_memory_window)
    
    def _weighted_soft_vote(
        self,
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float]
    ) -> tuple:
        """Compute weighted soft vote."""
        weighted_sum = 0.0
        weight_total = 0.0
        contributions = {}
        
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w <= 0:
                continue
            contribution = signal.action * signal.confidence * w
            contributions[agent_name] = contribution
            weighted_sum += contribution
            weight_total += w
        
        action = weighted_sum / weight_total if weight_total > 0 else 0.0
        return action, contributions
    
    def _check_conflict(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> tuple:
        """Detect agent conflicts."""
        directions = []
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0 and abs(signal.action) > 0.1:
                directions.append(np.sign(signal.action))
        
        if len(directions) < 2:
            return False, 0.0
        
        unique = set(directions)
        if len(unique) > 1:
            max_side = max(sum(1 for d in directions if d > 0), sum(1 for d in directions if d < 0))
            agreement = max_side / len(directions)
            if agreement < self.conflict_threshold:
                return True, (1 - agreement) * self.max_conflict_penalty
        
        return False, 0.0
    
    def _compute_momentum_memory(self) -> float:
        """Exponentially-weighted momentum from recent actions."""
        if len(self._action_history) < 3:
            return 0.0
        recent = list(self._action_history)
        weights = np.array([0.5 ** (len(recent) - 1 - i) for i in range(len(recent))])
        weights /= weights.sum()
        return np.dot(recent, weights)
    
    def _compute_confidence(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> float:
        conf_sum = 0.0
        w_sum = 0.0
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0:
                conf_sum += signal.confidence * w
                w_sum += w
        return conf_sum / w_sum if w_sum > 0 else 0.0
    
    def combine_signals(
        self,
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float],
        current_volatility: float = 0.15,
    ) -> EnsembleOutput:
        """Combine agent signals into alpha overlay signal."""
        
        if not signals:
            return EnsembleOutput(0.0, 0.0, False, 0.0, {})
        
        action, contributions = self._weighted_soft_vote(signals, weights)
        
        is_conflicting, penalty = self._check_conflict(signals, weights)
        if is_conflicting:
            action *= (1 - penalty)
        
        # Momentum memory: carry forward when current signal is weak
        momentum = self._compute_momentum_memory()
        if abs(action) < 0.15 and abs(momentum) > 0.2:
            action += momentum * self.momentum_carry_factor
        
        action = np.clip(action, -1.0, 1.0)
        self._action_history.append(action)
        
        confidence = self._compute_confidence(signals, weights)
        if is_conflicting:
            confidence *= 0.8
        
        return EnsembleOutput(action, confidence, is_conflicting, penalty, contributions)
