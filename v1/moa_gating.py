"""
Gating Network Module for MoA Trading Framework
Implements Top-K sparse gating to select the most relevant agents.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GatingOutput:
    """
    Output from the gating network.
    
    Attributes:
        weights: Dictionary of agent_name -> weight (only active agents included)
        active_agents: List of agent names that were selected
        regime_probs: Current regime probability distribution
        conflict_score: Estimated conflict between selected agents
    """
    weights: Dict[str, float]
    active_agents: List[str]
    regime_probs: Dict[str, float]
    conflict_score: float


class MoAGatingNetwork:
    """
    Gating Network for Mixture of Agents.
    
    Maps regime probabilities to agent weights using:
    - Regime-agent affinity matrix
    - Top-K sparse selection
    - Temperature-controlled sharpness
    """
    
    # Default regime-agent affinity matrix
    # Rows: regimes (Growth, Stagnation, Transition, Crisis)
    # Columns: agents (Trend, MeanReversion, Volatility, Crisis)
    DEFAULT_AFFINITY = np.array([
        [0.7, 0.2, 0.1, 0.0],   # Growth: mostly Trend
        [0.1, 0.7, 0.2, 0.0],   # Stagnation: mostly MeanReversion
        [0.2, 0.1, 0.6, 0.1],   # Transition: mostly Volatility
        [0.4, 0.0, 0.1, 0.5],   # Crisis: Trend (short) + Crisis agent
    ])
    
    REGIME_NAMES = ['Growth', 'Stagnation', 'Transition', 'Crisis']
    AGENT_NAMES = ['TrendAgent', 'MeanReversionAgent', 'VolatilityAgent', 'CrisisAgent']
    
    def __init__(
        self,
        top_k: int = 2,
        temperature: float = 1.0,
        min_weight: float = 0.1,
        affinity_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize the gating network.
        
        Args:
            top_k: Number of top agents to select
            temperature: Temperature for softmax (lower = sharper selection)
            min_weight: Minimum weight for selected agents
            affinity_matrix: Custom regime-agent affinity matrix
        """
        self.top_k = top_k
        self.temperature = temperature
        self.min_weight = min_weight
        
        if affinity_matrix is not None:
            self.affinity = affinity_matrix
        else:
            self.affinity = self.DEFAULT_AFFINITY.copy()
    
    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute softmax with temperature."""
        x = x / temperature
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _compute_raw_weights(self, regime_probs: Dict[str, float]) -> np.ndarray:
        """
        Compute raw agent weights from regime probabilities.
        
        weights = A^T @ regime_probs
        where A is the affinity matrix.
        """
        # Convert regime probs to array
        regime_vec = np.array([
            regime_probs.get('Growth', 0.25),
            regime_probs.get('Stagnation', 0.25),
            regime_probs.get('Transition', 0.25),
            regime_probs.get('Crisis', 0.25),
        ])
        
        # Matrix multiply: (4 agents) = (4 regimes x 4 agents)^T @ (4 regimes)
        raw_weights = self.affinity.T @ regime_vec
        
        return raw_weights
    
    def _apply_top_k(self, weights: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Apply Top-K selection.
        
        Returns:
            (sparse_weights, selected_indices)
        """
        # Get top-k indices
        top_indices = np.argsort(weights)[-self.top_k:]
        
        # Create sparse weights
        sparse_weights = np.zeros_like(weights)
        sparse_weights[top_indices] = weights[top_indices]
        
        # Renormalize
        if sparse_weights.sum() > 0:
            sparse_weights = sparse_weights / sparse_weights.sum()
        
        # Enforce minimum weight
        for i in top_indices:
            if sparse_weights[i] < self.min_weight and sparse_weights[i] > 0:
                sparse_weights[i] = self.min_weight
        
        # Renormalize again
        if sparse_weights.sum() > 0:
            sparse_weights = sparse_weights / sparse_weights.sum()
        
        return sparse_weights, list(top_indices)
    
    def _estimate_conflict(
        self, 
        selected_indices: List[int], 
        regime_probs: Dict[str, float]
    ) -> float:
        """
        Estimate potential conflict between selected agents.
        
        Higher conflict when agents favor opposite strategies.
        """
        if len(selected_indices) < 2:
            return 0.0
        
        # Define conflicting pairs (index pairs)
        conflicts = [
            (0, 1),  # Trend vs MeanReversion
            (2, 3),  # Volatility vs Crisis (both defensive but different)
        ]
        
        conflict_score = 0.0
        for i, j in conflicts:
            if i in selected_indices and j in selected_indices:
                # Both conflicting agents are selected
                conflict_score += 0.3
        
        # Additional conflict if regime probabilities are uncertain
        regime_vec = np.array([
            regime_probs.get('Growth', 0.25),
            regime_probs.get('Stagnation', 0.25),
            regime_probs.get('Transition', 0.25),
            regime_probs.get('Crisis', 0.25),
        ])
        
        # High entropy = high uncertainty
        entropy = -np.sum(regime_vec * np.log(regime_vec + 1e-8))
        max_entropy = np.log(4)  # Maximum entropy for 4 classes
        normalized_entropy = entropy / max_entropy
        
        conflict_score += normalized_entropy * 0.3
        
        return min(1.0, conflict_score)
    
    def compute_weights(self, regime_probs: Dict[str, float]) -> GatingOutput:
        """
        Compute agent weights from regime probabilities.
        
        Args:
            regime_probs: Dictionary with P(Growth), P(Stagnation), etc.
            
        Returns:
            GatingOutput with weights, active agents, and conflict score
        """
        # Compute raw weights
        raw_weights = self._compute_raw_weights(regime_probs)
        
        # Apply softmax with temperature
        soft_weights = self._softmax(raw_weights, self.temperature)
        
        # Apply Top-K selection
        sparse_weights, selected_indices = self._apply_top_k(soft_weights)
        
        # Convert to dictionary
        weights_dict = {}
        active_agents = []
        for i, agent_name in enumerate(self.AGENT_NAMES):
            if sparse_weights[i] > 0:
                weights_dict[agent_name] = sparse_weights[i]
                active_agents.append(agent_name)
        
        # Estimate conflict
        conflict_score = self._estimate_conflict(selected_indices, regime_probs)
        
        return GatingOutput(
            weights=weights_dict,
            active_agents=active_agents,
            regime_probs=regime_probs,
            conflict_score=conflict_score
        )
    
    def update_affinity(self, performance_feedback: Dict[str, float]) -> None:
        """
        Update affinity matrix based on performance feedback.
        
        This enables online learning of regime-agent mappings.
        
        Args:
            performance_feedback: Dictionary of agent_name -> recent performance
        """
        # Simple exponential moving average update
        learning_rate = 0.1
        
        for i, agent_name in enumerate(self.AGENT_NAMES):
            if agent_name in performance_feedback:
                perf = performance_feedback[agent_name]
                
                # Boost affinity for good-performing agents
                if perf > 0:
                    self.affinity[:, i] *= (1 + learning_rate * perf)
                else:
                    self.affinity[:, i] *= (1 + learning_rate * perf)  # Reduce
        
        # Renormalize rows
        for i in range(self.affinity.shape[0]):
            self.affinity[i] = self.affinity[i] / self.affinity[i].sum()


if __name__ == "__main__":
    # Test the gating network
    gating = MoAGatingNetwork(top_k=2)
    
    # Test with different regime scenarios
    scenarios = [
        {"Growth": 0.8, "Stagnation": 0.1, "Transition": 0.05, "Crisis": 0.05},
        {"Growth": 0.1, "Stagnation": 0.7, "Transition": 0.15, "Crisis": 0.05},
        {"Growth": 0.1, "Stagnation": 0.1, "Transition": 0.7, "Crisis": 0.1},
        {"Growth": 0.1, "Stagnation": 0.05, "Transition": 0.1, "Crisis": 0.75},
        {"Growth": 0.25, "Stagnation": 0.25, "Transition": 0.25, "Crisis": 0.25},  # Uncertain
    ]
    
    print("Gating Network Tests:")
    print("=" * 60)
    
    for i, regime_probs in enumerate(scenarios):
        output = gating.compute_weights(regime_probs)
        dominant = max(regime_probs, key=regime_probs.get)
        print(f"\nScenario {i+1}: Dominant regime = {dominant}")
        print(f"  Active Agents: {output.active_agents}")
        print(f"  Weights: {output.weights}")
        print(f"  Conflict Score: {output.conflict_score:.3f}")
