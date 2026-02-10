"""
V3 MoA Soft Ensemble — With momentum memory.

Key V3 changes over V2:
- Momentum memory: rolling window of recent ensemble actions.
  If momentum memory is bullish and current action is near-zero,
  carry forward a decayed bullish bias.
- Agreement floor raised from 0.30 → 0.40
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
    """Output from the ensemble combination."""
    final_action: float
    confidence: float
    is_conflicting: bool
    conflict_penalty: float
    agent_contributions: Dict[str, float]


class MoASoftEnsemble:
    """
    V3 Soft ensemble with momentum memory.
    
    Now carries forward bullish/bearish momentum from recent bars
    to reduce time spent near-zero.
    """
    
    def __init__(
        self,
        conflict_threshold: float = 0.8,
        max_conflict_penalty: float = 0.4,
        agreement_floor: float = 0.40,           # V3: 0.30 → 0.40
        min_agreement_agents: int = 2,
        min_agreement_confidence: float = 0.5,
        momentum_memory_window: int = 10,         # V3 new
        momentum_carry_factor: float = 0.5,       # V3 new
    ):
        self.conflict_threshold = conflict_threshold
        self.max_conflict_penalty = max_conflict_penalty
        self.agreement_floor = agreement_floor
        self.min_agreement_agents = min_agreement_agents
        self.min_agreement_confidence = min_agreement_confidence
        self.momentum_memory_window = momentum_memory_window
        self.momentum_carry_factor = momentum_carry_factor
        
        # V3: Track recent ensemble actions for momentum memory
        self._action_history: deque = deque(maxlen=momentum_memory_window)
    
    def _weighted_soft_vote(
        self,
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float]
    ) -> tuple:
        """Compute weighted soft vote — no regime_fit dampening (same as V2)."""
        weighted_sum = 0.0
        weight_total = 0.0
        contributions = {}
        
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w <= 0:
                continue
            
            # V2/V3: No regime_fit multiplication
            contribution = signal.action * signal.confidence * w
            contributions[agent_name] = contribution
            weighted_sum += contribution
            weight_total += w
        
        if weight_total > 0:
            action = weighted_sum / weight_total
        else:
            action = 0.0
        
        return action, contributions
    
    def _check_conflict(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> tuple:
        """Detect agent conflicts (same as V2)."""
        directions = []
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0 and abs(signal.action) > 0.1:
                directions.append(np.sign(signal.action))
        
        if len(directions) < 2:
            return False, 0.0
        
        unique_directions = set(directions)
        if len(unique_directions) > 1:
            positive = sum(1 for d in directions if d > 0)
            negative = sum(1 for d in directions if d < 0)
            
            max_side = max(positive, negative)
            agreement_ratio = max_side / len(directions)
            
            if agreement_ratio < self.conflict_threshold:
                penalty = (1 - agreement_ratio) * self.max_conflict_penalty
                return True, penalty
        
        return False, 0.0
    
    def _check_agreement(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> tuple:
        """Check if multiple agents agree (floor enforcement)."""
        bullish_count = 0
        bearish_count = 0
        
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0 and signal.confidence >= self.min_agreement_confidence:
                if signal.action > 0.1:
                    bullish_count += 1
                elif signal.action < -0.1:
                    bearish_count += 1
        
        if bullish_count >= self.min_agreement_agents:
            return True, 1  # Bullish agreement
        elif bearish_count >= self.min_agreement_agents:
            return True, -1  # Bearish agreement
        
        return False, 0
    
    def _compute_momentum_memory(self) -> float:
        """
        V3 NEW: Compute momentum memory from recent ensemble actions.
        Returns a decayed bias based on recent action direction.
        """
        if len(self._action_history) < 3:
            return 0.0
        
        recent = list(self._action_history)
        # Exponentially weighted mean — recent actions matter more
        weights = np.array([0.5 ** (len(recent) - 1 - i) for i in range(len(recent))])
        weights /= weights.sum()
        
        momentum = np.dot(recent, weights)
        return momentum
    
    def _compute_confidence(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> float:
        """Weighted average confidence."""
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
        """Combine agent signals with momentum memory."""
        
        if not signals:
            return EnsembleOutput(
                final_action=0.0, confidence=0.0,
                is_conflicting=False, conflict_penalty=0.0,
                agent_contributions={}
            )
        
        # Weighted soft vote
        action, contributions = self._weighted_soft_vote(signals, weights)
        
        # Conflict check
        is_conflicting, penalty = self._check_conflict(signals, weights)
        if is_conflicting:
            action *= (1 - penalty)
        
        # Agreement floor
        has_agreement, agreement_dir = self._check_agreement(signals, weights)
        if has_agreement:
            if abs(action) < self.agreement_floor:
                action = agreement_dir * max(abs(action), self.agreement_floor)
        
        # V3 NEW: Momentum memory
        momentum = self._compute_momentum_memory()
        if abs(action) < 0.15 and abs(momentum) > 0.2:
            # Current signal is weak, but recent history had direction
            carry = momentum * self.momentum_carry_factor
            action = action + carry
        
        # Clamp
        action = np.clip(action, -1.0, 1.0)
        
        # Record action for momentum memory
        self._action_history.append(action)
        
        # Confidence
        confidence = self._compute_confidence(signals, weights)
        if is_conflicting:
            confidence *= 0.8
        
        return EnsembleOutput(
            final_action=action,
            confidence=confidence,
            is_conflicting=is_conflicting,
            conflict_penalty=penalty,
            agent_contributions=contributions
        )
