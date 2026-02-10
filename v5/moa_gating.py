"""
V3 MoA Gating Network — Trend confirmation bonus.

Key V3 change over V2:
- When ExponentialMomentumAgent and TrendAgent both selected,
  boost their combined weight by 1.2× (trend confirmation bonus)
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
    """V3 Gating Network with trend confirmation bonus."""
    
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
        trend_confirmation_bonus: float = 1.2,  # V3 new
    ):
        self.top_k = top_k
        self.temperature = temperature
        self.trend_confirmation_bonus = trend_confirmation_bonus
        
        # Same affinity matrix as V2
        self.affinity_matrix = np.array([
            [0.80, 0.20, 0.40, 0.60],  # TrendAgent
            [0.10, 0.90, 0.30, 0.05],  # MeanReversionAgent
            [0.30, 0.30, 0.85, 0.20],  # VolatilityAgent
            [0.10, 0.05, 0.20, 0.90],  # CrisisAgent
            [0.95, 0.15, 0.50, 0.40],  # ExponentialMomentumAgent
        ])
    
    def _compute_raw_weights(self, regime_probs: Dict[str, float]) -> np.ndarray:
        prob_vector = np.array([regime_probs.get(r, 0) for r in self.REGIME_NAMES])
        return self.affinity_matrix @ prob_vector
    
    def _softmax(self, x: np.ndarray, temperature: float) -> np.ndarray:
        x_scaled = x / max(temperature, 0.01)
        exp_x = np.exp(x_scaled - np.max(x_scaled))
        return exp_x / exp_x.sum()
    
    def _apply_top_k(self, weights: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        k = min(self.top_k, len(weights))
        top_indices = np.argsort(weights)[-k:]
        sparse_weights = np.zeros_like(weights)
        sparse_weights[top_indices] = weights[top_indices]
        total = sparse_weights.sum()
        if total > 0:
            sparse_weights /= total
        return sparse_weights, sorted(top_indices.tolist())
    
    def _apply_trend_confirmation_bonus(
        self, 
        weights: np.ndarray, 
        selected_indices: List[int]
    ) -> np.ndarray:
        """
        V3 NEW: If both TrendAgent (idx 0) and ExponentialMomentumAgent (idx 4)
        are selected, boost their weights by trend_confirmation_bonus.
        """
        trend_idx = 0
        momentum_idx = 4
        
        if trend_idx in selected_indices and momentum_idx in selected_indices:
            weights[trend_idx] *= self.trend_confirmation_bonus
            weights[momentum_idx] *= self.trend_confirmation_bonus
            # Renormalize
            total = weights.sum()
            if total > 0:
                weights /= total
        
        return weights
    
    def _estimate_conflict(self, selected_indices: List[int], regime_probs: Dict[str, float]) -> float:
        if len(selected_indices) < 2:
            return 0.0
        affinities = self.affinity_matrix[selected_indices]
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
        raw_weights = self._compute_raw_weights(regime_probs)
        soft_weights = self._softmax(raw_weights, self.temperature)
        sparse_weights, selected_indices = self._apply_top_k(soft_weights)
        
        # V3: Apply trend confirmation bonus
        sparse_weights = self._apply_trend_confirmation_bonus(sparse_weights, selected_indices)
        
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
