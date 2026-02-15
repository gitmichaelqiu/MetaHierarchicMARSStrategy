"""
CIO Risk Manager — Portfolio-level risk management.

Type: risk_manager

Three sub-components:
1. Drawdown Guard — scale positions when portfolio DD is high
2. Correlation Monitor — reduce exposure when stocks move together
3. Cash Reserve Logic — dynamic cash allocation based on VIX/regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from base_model_adapter import BaseModelSignal


@dataclass
class RiskOutput:
    """Risk manager output — constraints on portfolio allocation."""
    max_exposure: float        # Maximum invested fraction (1.0 - min_cash)
    position_scalar: float     # Multiplier on all positions
    risk_flags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class RiskManager:
    """
    Portfolio-level risk management with three defense layers.
    """
    
    def __init__(
        self,
        # Drawdown thresholds
        dd_caution: float = 0.08,
        dd_danger: float = 0.15,
        # Correlation thresholds
        corr_caution: float = 0.60,
        corr_danger: float = 0.80,
        # Cash reserve
        min_cash: float = 0.05,
        max_cash: float = 0.40,
        # VIX thresholds
        vix_caution: float = 22.0,
        vix_fear: float = 32.0,
    ):
        self.dd_caution = dd_caution
        self.dd_danger = dd_danger
        self.corr_caution = corr_caution
        self.corr_danger = corr_danger
        self.min_cash = min_cash
        self.max_cash = max_cash
        self.vix_caution = vix_caution
        self.vix_fear = vix_fear
        
        # State
        self._equity_curve: list = [1.0]
        self._peak_equity: float = 1.0
    
    def update_equity(self, portfolio_return: float):
        new_eq = self._equity_curve[-1] * (1 + portfolio_return)
        self._equity_curve.append(new_eq)
        self._peak_equity = max(self._peak_equity, new_eq)
    
    @property
    def current_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        return max(0, 1 - self._equity_curve[-1] / self._peak_equity)
    
    def assess_risk(
        self,
        signals: Dict[str, BaseModelSignal],
        vix_value: Optional[float] = None,
        avg_correlation: Optional[float] = None,
    ) -> RiskOutput:
        """
        Compute portfolio risk constraints.
        
        Returns max_exposure and position_scalar that the CIO
        applies after allocation.
        """
        flags = []
        cash_target = self.min_cash
        scalar = 1.0
        
        # ── Layer 1: Drawdown Guard ──
        dd = self.current_drawdown
        if dd > self.dd_danger:
            scalar *= 0.5
            cash_target = max(cash_target, 0.30)
            flags.append(f"DD_DANGER ({dd:.1%})")
        elif dd > self.dd_caution:
            factor = 1.0 - 0.5 * (dd - self.dd_caution) / (self.dd_danger - self.dd_caution)
            scalar *= factor
            cash_target = max(cash_target, 0.15)
            flags.append(f"DD_CAUTION ({dd:.1%})")
        
        # ── Layer 2: Correlation Monitor ──
        if avg_correlation is not None and not np.isnan(avg_correlation):
            if avg_correlation > self.corr_danger:
                cash_target = max(cash_target, 0.30)
                scalar *= 0.7
                flags.append(f"CORR_DANGER ({avg_correlation:.2f})")
            elif avg_correlation > self.corr_caution:
                cash_target = max(cash_target, 0.15)
                flags.append(f"CORR_CAUTION ({avg_correlation:.2f})")
        
        # ── Layer 3: VIX / Cash Reserve ──
        if vix_value is not None and not np.isnan(vix_value):
            if vix_value > self.vix_fear:
                cash_target = max(cash_target, self.max_cash)
                scalar *= 0.6
                flags.append(f"VIX_FEAR ({vix_value:.1f})")
            elif vix_value > self.vix_caution:
                vix_cash = 0.10 + 0.20 * (vix_value - self.vix_caution) / (self.vix_fear - self.vix_caution)
                cash_target = max(cash_target, vix_cash)
                flags.append(f"VIX_CAUTION ({vix_value:.1f})")
        
        # ── Layer 4: Regime-based ──
        crisis_count = sum(1 for s in signals.values() if s.regime == 'Crisis')
        crisis_pct = crisis_count / max(1, len(signals))
        if crisis_pct > 0.5:
            cash_target = max(cash_target, 0.35)
            scalar *= 0.6
            flags.append(f"MAJORITY_CRISIS ({crisis_count}/{len(signals)})")
        elif crisis_pct > 0.40:
            cash_target = max(cash_target, 0.12)
            flags.append(f"SOME_CRISIS ({crisis_count}/{len(signals)})")
        
        cash_target = min(cash_target, self.max_cash)
        max_exposure = 1.0 - cash_target
        
        return RiskOutput(
            max_exposure=max_exposure,
            position_scalar=scalar,
            risk_flags=flags,
            metadata={
                'drawdown': dd,
                'cash_target': cash_target,
                'vix': vix_value,
                'avg_correlation': avg_correlation,
                'crisis_pct': crisis_pct,
            }
        )
    
    def reset(self):
        self._equity_curve = [1.0]
        self._peak_equity = 1.0
