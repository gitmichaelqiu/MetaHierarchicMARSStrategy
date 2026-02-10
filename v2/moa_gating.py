"""
V2 MoA Gating Network
Redesigned to support 5 agents with Top-K=3 selection.

Key V2 improvements over V1:
- 5 agents (added ExponentialMomentumAgent)  
- Top-K=3 allows momentum agent to coexist with trend agent
- Updated affinity matrix: ExponentialMomentumAgent has highest Growth affinity
- Avoids selecting conflicting agents (Trend + MeanReversion during Growth)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GatingOutput:
    """Output from the gating network."""
    weights: Dict[str, float]
    active_agents: List[str]
    regime_probs: Dict[str, float]
    conflict_score: float


class MoAGatingNetwork:
    """
    V2 Gating Network with 5-agent support and Top-K=3 selection.
    """
    
    AGENT_NAMES = [
        'TrendAgent', 
        'MeanReversionAgent', 
        'VolatilityAgent', 
        'CrisisAgent',
        'ExponentialMomentumAgent'
    ]
    
    REGIME_NAMES = ['Growth', 'Stagnation', 'Transition', 'Crisis']
    
    def __init__(
        self,
        top_k: int = 3,
        temperature: float = 0.8,
    ):
        self.top_k = top_k
        self.temperature = temperature
        
        # V2 Affinity Matrix: 5 agents x 4 regimes
        # Rows: agents, Columns: Growth, Stagnation, Transition, Crisis
        self.affinity_matrix = np.array([
            # Growth  Stag  Trans  Crisis
            [0.80,   0.20,  0.40,  0.60],  # TrendAgent
            [0.10,   0.90,  0.30,  0.05],  # MeanReversionAgent
            [0.30,   0.30,  0.85,  0.20],  # VolatilityAgent
            [0.10,   0.05,  0.20,  0.90],  # CrisisAgent
            [0.95,   0.15,  0.50,  0.40],  # ExponentialMomentumAgent (new)
        ])
    
    def _compute_raw_weights(self, regime_probs: Dict[str, float]) -> np.ndarray:
        """Compute raw agent weights from regime probabilities."""
        prob_vector = np.array([
            regime_probs.get(r, 0) for r in self.REGIME_NAMES
        ])
        
        raw_weights = self.affinity_matrix @ prob_vector
        return raw_weights
    
    def _softmax(self, x: np.ndarray, temperature: float) -> np.ndarray:
        """Temperature-scaled softmax."""
        x_scaled = x / max(temperature, 0.01)
        exp_x = np.exp(x_scaled - np.max(x_scaled))
        return exp_x / exp_x.sum()
    
    def _apply_top_k(self, weights: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Select top-K agents and renormalize."""
        k = min(self.top_k, len(weights))
        top_indices = np.argsort(weights)[-k:]
        
        sparse_weights = np.zeros_like(weights)
        sparse_weights[top_indices] = weights[top_indices]
        
        total = sparse_weights.sum()
        if total > 0:
            sparse_weights /= total
        
        return sparse_weights, sorted(top_indices.tolist())
    
    def _estimate_conflict(
        self, 
        selected_indices: List[int],
        regime_probs: Dict[str, float]
    ) -> float:
        """Estimate conflict between selected agents."""
        if len(selected_indices) < 2:
            return 0.0
        
        affinities = self.affinity_matrix[selected_indices]
        
        # Calculate pairwise cosine distance
        conflicts = []
        for i in range(len(affinities)):
            for j in range(i + 1, len(affinities)):
                dot = np.dot(affinities[i], affinities[j])
                norm_i = np.linalg.norm(affinities[i])
                norm_j = np.linalg.norm(affinities[j])
                if norm_i > 0 and norm_j > 0:
                    similarity = dot / (norm_i * norm_j)
                    conflicts.append(1 - similarity)
        
        return np.mean(conflicts) if conflicts else 0.0
    
    def compute_weights(self, regime_probs: Dict[str, float]) -> GatingOutput:
        """
        Compute agent weights from regime probabilities.
        
        Returns:
            GatingOutput with sparse weights for top-K agents
        """
        raw_weights = self._compute_raw_weights(regime_probs)
        soft_weights = self._softmax(raw_weights, self.temperature)
        sparse_weights, selected_indices = self._apply_top_k(soft_weights)
        
        weights_dict = {}
        active_agents = []
        for idx in selected_indices:
            if sparse_weights[idx] > 0:
                name = self.AGENT_NAMES[idx]
                weights_dict[name] = float(sparse_weights[idx])
                active_agents.append(name)
        
        conflict_score = self._estimate_conflict(selected_indices, regime_probs)
        
        return GatingOutput(
            weights=weights_dict,
            active_agents=active_agents,
            regime_probs=regime_probs,
            conflict_score=conflict_score
        )
