"""
Soft-Ensemble Module for MoA Trading Framework
Implements weighted soft voting with conflict resolution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from agents.base_agent import BaseAgent, AgentSignal


@dataclass
class EnsembleOutput:
    """
    Output from the ensemble voting.
    
    Attributes:
        final_action: Combined action in [-1, 1]
        raw_action: Action before conflict adjustment
        confidence: Overall confidence in the signal
        is_conflicting: Whether agents significantly disagree
        conflict_penalty: Penalty applied due to conflict
        agent_contributions: Each agent's contribution to final action
    """
    final_action: float
    raw_action: float
    confidence: float
    is_conflicting: bool
    conflict_penalty: float
    agent_contributions: Dict[str, float]


class MoASoftEnsemble:
    """
    Soft-Ensemble Voting with Conflict Resolution.
    
    Features:
    - Weighted soft voting based on gating weights
    - Fuzzy conflict detection
    - Position reduction during high conflict
    - Transaction cost awareness
    """
    
    def __init__(
        self,
        conflict_threshold: float = 0.5,
        max_conflict_penalty: float = 0.7,
        volatility_scaling: bool = True,
        use_fuzzy_logic: bool = True
    ):
        """
        Initialize the ensemble.
        
        Args:
            conflict_threshold: Action difference to trigger conflict detection
            max_conflict_penalty: Maximum position reduction during conflict
            volatility_scaling: Scale down positions in high volatility
            use_fuzzy_logic: Use fuzzy logic for conflict resolution
        """
        self.conflict_threshold = conflict_threshold
        self.max_conflict_penalty = max_conflict_penalty
        self.volatility_scaling = volatility_scaling
        self.use_fuzzy_logic = use_fuzzy_logic
    
    def _detect_conflict(
        self, 
        signals: Dict[str, AgentSignal], 
        weights: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        Detect conflict between agent signals.
        
        Returns:
            (is_conflicting, conflict_score)
        """
        if len(signals) < 2:
            return False, 0.0
        
        # Get weighted actions
        weighted_actions = []
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0:
                weighted_actions.append((signal.action, w))
        
        if len(weighted_actions) < 2:
            return False, 0.0
        
        # Check for sign disagreement among weighted agents
        actions = [a for a, w in weighted_actions]
        has_long = any(a > self.conflict_threshold for a in actions)
        has_short = any(a < -self.conflict_threshold for a in actions)
        
        if has_long and has_short:
            # Calculate conflict severity
            max_long = max(a for a in actions if a > 0) if has_long else 0
            max_short = min(a for a in actions if a < 0) if has_short else 0
            conflict_score = (max_long - max_short) / 2
            return True, min(1.0, conflict_score)
        
        # Check for high variance in actions
        action_std = np.std(actions)
        if action_std > self.conflict_threshold:
            return True, min(1.0, action_std)
        
        return False, 0.0
    
    def _compute_soft_vote(
        self, 
        signals: Dict[str, AgentSignal], 
        weights: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted soft vote.
        
        Returns:
            (combined_action, agent_contributions)
        """
        weighted_sum = 0.0
        weight_total = 0.0
        contributions = {}
        
        for agent_name, signal in signals.items():
            w = weights.get(agent_name, 0)
            if w > 0:
                # Weight by gating weight, confidence, and regime fit
                effective_weight = w * signal.confidence * signal.regime_fit
                contribution = signal.action * effective_weight
                
                weighted_sum += contribution
                weight_total += effective_weight
                contributions[agent_name] = contribution
        
        if weight_total < 1e-6:
            return 0.0, contributions
        
        combined = weighted_sum / weight_total
        return combined, contributions
    
    def _apply_fuzzy_conflict_resolution(
        self, 
        action: float, 
        conflict_score: float,
        volatility: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Apply fuzzy logic conflict resolution.
        
        Rule: IF Conflict is High AND Volatility is High THEN Position = Low
        
        Returns:
            (adjusted_action, penalty_applied)
        """
        if not self.use_fuzzy_logic:
            return action, 0.0
        
        # Base penalty from conflict
        penalty = conflict_score * self.max_conflict_penalty
        
        # Increase penalty if volatility is also high
        if volatility is not None and volatility > 0.30:  # 30% annualized
            vol_factor = min(1.0, (volatility - 0.30) / 0.30)
            penalty = penalty * (1 + vol_factor * 0.5)
        
        penalty = min(self.max_conflict_penalty, penalty)
        
        # Apply penalty (reduce position size)
        adjusted_action = action * (1 - penalty)
        
        return adjusted_action, penalty
    
    def _compute_ensemble_confidence(
        self, 
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float],
        conflict_score: float
    ) -> float:
        """
        Compute overall ensemble confidence.
        """
        if not signals:
            return 0.0
        
        # Weighted average of individual confidences
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
        
        # Reduce confidence during conflict
        ensemble_confidence = avg_confidence * (1 - conflict_score * 0.5)
        
        return ensemble_confidence
    
    def combine_signals(
        self,
        signals: Dict[str, AgentSignal],
        weights: Dict[str, float],
        current_volatility: Optional[float] = None
    ) -> EnsembleOutput:
        """
        Combine agent signals into a single trading action.
        
        Args:
            signals: Dictionary of agent_name -> AgentSignal
            weights: Dictionary of agent_name -> weight from gating network
            current_volatility: Current annualized volatility (optional)
            
        Returns:
            EnsembleOutput with final action and metadata
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
        
        # Detect conflict
        is_conflicting, conflict_score = self._detect_conflict(signals, weights)
        
        # Compute soft vote
        raw_action, contributions = self._compute_soft_vote(signals, weights)
        
        # Apply conflict resolution
        if is_conflicting:
            final_action, penalty = self._apply_fuzzy_conflict_resolution(
                raw_action, conflict_score, current_volatility
            )
        else:
            final_action = raw_action
            penalty = 0.0
        
        # Optional volatility scaling
        if self.volatility_scaling and current_volatility is not None:
            if current_volatility > 0.40:  # High volatility
                vol_scale = max(0.3, 1.0 - (current_volatility - 0.40) / 0.40)
                final_action *= vol_scale
        
        # Compute confidence
        confidence = self._compute_ensemble_confidence(signals, weights, conflict_score)
        
        # Clamp final action
        final_action = np.clip(final_action, -1.0, 1.0)
        
        return EnsembleOutput(
            final_action=final_action,
            raw_action=raw_action,
            confidence=confidence,
            is_conflicting=is_conflicting,
            conflict_penalty=penalty,
            agent_contributions=contributions
        )


if __name__ == "__main__":
    # Test the ensemble
    from agents.base_agent import AgentSignal
    
    ensemble = MoASoftEnsemble()
    
    # Scenario 1: Agreement
    signals_agree = {
        'TrendAgent': AgentSignal(action=0.8, confidence=0.9, regime_fit=0.8),
        'VolatilityAgent': AgentSignal(action=0.6, confidence=0.7, regime_fit=0.6),
    }
    weights_agree = {'TrendAgent': 0.6, 'VolatilityAgent': 0.4}
    
    result = ensemble.combine_signals(signals_agree, weights_agree)
    print("Scenario 1: Agreement")
    print(f"  Final Action: {result.final_action:.3f}")
    print(f"  Is Conflicting: {result.is_conflicting}")
    print(f"  Confidence: {result.confidence:.3f}")
    
    # Scenario 2: Conflict
    signals_conflict = {
        'TrendAgent': AgentSignal(action=0.9, confidence=0.8, regime_fit=0.7),
        'MeanReversionAgent': AgentSignal(action=-0.7, confidence=0.8, regime_fit=0.6),
    }
    weights_conflict = {'TrendAgent': 0.5, 'MeanReversionAgent': 0.5}
    
    result = ensemble.combine_signals(signals_conflict, weights_conflict, current_volatility=0.35)
    print("\nScenario 2: Conflict")
    print(f"  Raw Action: {result.raw_action:.3f}")
    print(f"  Final Action: {result.final_action:.3f}")
    print(f"  Is Conflicting: {result.is_conflicting}")
    print(f"  Conflict Penalty: {result.conflict_penalty:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
