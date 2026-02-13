"""
Allocator Base — Abstract interface for all allocation strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from base_model_adapter import BaseModelSignal


@dataclass
class AllocationOutput:
    """Output from an allocator."""
    weights: Dict[str, float]      # ticker -> weight (0 to 1, sum ≤ 1)
    cash_weight: float             # remaining cash fraction
    metadata: Dict = field(default_factory=dict)


class BaseAllocator(ABC):
    """Abstract base for all allocators."""
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Allocator version tag, e.g. 'v1-risk-parity'."""
        ...
    
    @abstractmethod
    def allocate(
        self,
        signals: Dict[str, BaseModelSignal],
        max_exposure: float = 1.0,
    ) -> AllocationOutput:
        """
        Compute allocation weights across tickers.
        
        Args:
            signals: Per-ticker signals from base models
            max_exposure: Maximum total weight (1.0 - cash_reserve)
            
        Returns:
            AllocationOutput with normalized weights and cash
        """
        ...
