"""
V2 Soft-Ensemble Module for MoA Trading Framework
Simplified ensemble with less dampening.

Key V2 improvements over V1:
- Removed regime_fit multiplication from soft vote (already handled by gating)
- Added minimum position floor when agents agree on direction
- Higher conflict threshold (0.8 vs 0.5) — only penalize strong disagreements
- Reduced max conflict penalty (0.4 vs 0.7)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V1_DIR = os.path.join(_PROJECT_ROOT, 'v1')
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)

from agents.base_agent import AgentSignal


@dataclass
class EnsembleOutput:
    """Output from the ensemble voting."""
    final_action: float
    raw_action: float
    confidence: float
    is_conflicting: bool
    conflict_penalty: float
    agent_contributions: Dict[str, float]


class MoASoftEnsemble:
    """
    V2 Soft-Ensemble: Less dampening, stronger signals.
    """
    
    def __init__(
        self,
        conflict_threshold: float = 0.8,       # V2: higher (was 0.5)
        max_conflict_penalty: float = 0.4,      # V2: lower (was 0.7)
        min_agreement_floor: float = 0.30,      # V2: NEW — minimum position when agents agree
    ):
        self.conflict_threshold = conflict_threshold
        self.max_conflict_penalty = max_conflict_penalty
        self.min_agreement_floor = min_agreement_floor
    
    def _detect_conflict(
        self, 
        signals: Dict[str, AgentSignal], 
        weights: Dict[str, float]
    ) -> Tuple[bool, float]:
        """Detect conflict between agent signals."""
        if len(signals) < 2:
            return False, 0.0
        
        actions = []
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0:
                actions.append(signal.action)
        
        if len(actions) < 2:
            return False, 0.0
        
        # Only flag conflict for STRONG disagreement
        has_long = any(a > self.conflict_threshold for a in actions)
        has_short = any(a < -self.conflict_threshold for a in actions)
        
        if has_long and has_short:
            max_long = max(a for a in actions if a > 0)
            max_short = min(a for a in actions if a < 0)
            conflict_score = (max_long - max_short) / 2
            return True, min(1.0, conflict_score)
        
        return False, 0.0
    
    def _compute_soft_vote(
        self, 
        signals: Dict[str, AgentSignal], 
        weights: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        V2: Weighted soft vote WITHOUT regime_fit multiplication.
        Regime suitability is already handled by the gating weights.
        """
        weighted_sum = 0.0
        weight_total = 0.0
        contributions = {}
        
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0:
                # V2: Use weight * confidence only (no regime_fit dampening)
                effective_weight = w * signal.confidence
                contribution = signal.action * effective_weight
                
                weighted_sum += contribution
                weight_total += effective_weight
                contributions[agent_name] = contribution
        
        if weight_total < 1e-6:
            return 0.0, contributions
        
        combined = weighted_sum / weight_total
        return combined, contributions
    
    def _apply_agreement_floor(
        self,
        action: float,
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float]
    ) -> float:
        """
        V2 NEW: If ≥2 agents agree on direction with confidence > 0.4,
        enforce a minimum position magnitude.
        """
        active_signals = [
            s for name, s in signals.items() 
            if weights.get(name, 0) > 0 and s.confidence > 0.4
        ]
        
        if len(active_signals) < 2:
            return action
        
        # Count directional agreement
        long_count = sum(1 for s in active_signals if s.action > 0.2)
        short_count = sum(1 for s in active_signals if s.action < -0.2)
        
        if long_count >= 2 and action > 0:
            action = max(action, self.min_agreement_floor)
        elif short_count >= 2 and action < 0:
            action = min(action, -self.min_agreement_floor)
        
        return action
    
    def _compute_ensemble_confidence(
        self, 
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float],
        conflict_score: float
    ) -> float:
        """Compute overall ensemble confidence."""
        if not signals:
            return 0.0
        
        weighted_conf_sum = 0.0
        weight_total = 0.0
        
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0:
                weighted_conf_sum += signal.confidence * w
                weight_total += w
        
        if weight_total < 1e-6:
            return 0.0
        
        avg_confidence = weighted_conf_sum / weight_total
        
        # V2: Lighter conflict penalty on confidence (0.3 vs 0.5)
        ensemble_confidence = avg_confidence * (1 - conflict_score * 0.3)
        
        return ensemble_confidence
    
    def combine_signals(
        self,
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float],
        current_volatility: Optional[float] = None
    ) -> EnsembleOutput:
        """
        Combine agent signals into a single trading action.
        V2: Less dampening, agreement floor, lighter conflict penalty.
        """
        if not signals:
            return EnsembleOutput(
                final_action=0.0,
                raw_action=0.0,
                confidence=0.0,
                is_conflicting=False,
                conflict_penalty=0.0,
                agent_contributions={}
            )
        
        # Detect conflict (higher threshold in V2)
        is_conflicting, conflict_score = self._detect_conflict(signals, weights)
        
        # Compute soft vote (no regime_fit dampening)
        raw_action, contributions = self._compute_soft_vote(signals, weights)
        
        # Apply conflict resolution if needed
        if is_conflicting:
            penalty = min(self.max_conflict_penalty, conflict_score * self.max_conflict_penalty)
            final_action = raw_action * (1 - penalty)
        else:
            final_action = raw_action
            penalty = 0.0
        
        # V2: Apply agreement floor (boost weak signals when agents agree)
        final_action = self._apply_agreement_floor(final_action, signals, weights)
        
        # V2: NO volatility scaling here (let MetaController handle risk)
        
        # Compute confidence
        confidence = self._compute_ensemble_confidence(signals, weights, conflict_score)
        
        # Clamp
        final_action = np.clip(final_action, -1.0, 1.0)
        
        return EnsembleOutput(
            final_action=final_action,
            raw_action=raw_action,
            confidence=confidence,
            is_conflicting=is_conflicting,
            conflict_penalty=penalty,
            agent_contributions=contributions
        )
