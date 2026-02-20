"""
V7 Adapter — Wraps MoAStrategyV7 into the CIO BaseModelAdapter protocol.

This is the reference implementation showing how to adapt an existing
base model for the CIO framework.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V7_DIR = os.path.join(PROJECT_ROOT, 'v7')
V7_IMP_DIR = os.path.join(PROJECT_ROOT, 'v7_improved')
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')

for p in [V1_DIR, PROJECT_ROOT, V7_DIR, V7_IMP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from base_model_adapter import BaseModelAdapter, BaseModelSignal

# Import V7 Improved components
from v7_improved.regime_detector import RegimeDetector
from v7_improved.moa_gating import MoAGatingNetwork
from v7_improved.moa_ensemble import MoASoftEnsemble
from v7_improved.meta_controller import MetaController

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.volatility_agent import VolatilityAgent
from agents.crisis_agent import CrisisAgent
from v7_improved.v7_agents.exponential_momentum_agent import ExponentialMomentumAgent


class V7Adapter:
    """Wraps V7 MoA into the BaseModelAdapter protocol."""
    
    @property
    def model_name(self) -> str:
        return "V7-Improved-CrossAsset"
    
    def __init__(self):
        self._ticker: str = ""
        self._regime_detector: Optional[RegimeDetector] = None
        self._agents: Dict = {}
        self._gating: Optional[MoAGatingNetwork] = None
        self._ensemble: Optional[MoASoftEnsemble] = None
        self._controller: Optional[MetaController] = None
        
        # Cross-asset data (set via kwargs in initialize)
        self._vix_data = None
        self._vix_term_ratio = None
        self._weekly_trend = None
        self._sector_trend = None
        self._avg_correlation = None
    
    def initialize(self, ticker: str, df: pd.DataFrame, **kwargs) -> None:
        self._ticker = ticker
        self._regime_detector = RegimeDetector()
        self._agents = {
            'TrendAgent': TrendAgent(),
            'MeanReversionAgent': MeanReversionAgent(),
            'VolatilityAgent': VolatilityAgent(),
            'CrisisAgent': CrisisAgent(),
            'ExponentialMomentumAgent': ExponentialMomentumAgent(),
        }
        self._gating = MoAGatingNetwork(top_k=3, temperature=0.8)
        self._ensemble = MoASoftEnsemble()
        self._controller = MetaController()
        
        # Fit regime detector
        self._regime_detector.fit(df)
        
        # Accept cross-asset data
        self._vix_data = kwargs.get('vix_data')
        self._vix_term_ratio = kwargs.get('vix_term_ratio')
        self._weekly_trend = kwargs.get('weekly_trend')
        self._sector_trend = kwargs.get('sector_trend')
        self._avg_correlation = kwargs.get('avg_correlation')
    
    def _get_value(self, series, date):
        if series is None or (isinstance(series, pd.Series) and series.empty):
            return None
        try:
            if hasattr(date, 'tz') and date.tz is not None:
                date = date.tz_convert(None)
            idx = series.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx = idx.tz_convert(None)
            date_only = pd.Timestamp(date).normalize()
            mask = idx.normalize() <= date_only
            if mask.any():
                return float(series.iloc[idx[mask].get_loc(idx[mask][-1])])
        except Exception:
            pass
        return None
    
    def get_signal(self, df: pd.DataFrame, idx: int, portfolio_drawdown: Optional[float] = None) -> BaseModelSignal:
        current_data = df.iloc[:idx + 1]
        
        if len(current_data) < 60:
            return BaseModelSignal(
                ticker=self._ticker, position=0.0, confidence=0.0,
                regime='Unknown', regime_probs={}, volatility=0.2
            )
        
        regime_name, regime_probs = self._regime_detector.get_current_regime(current_data)
        if not regime_probs:
            return BaseModelSignal(
                ticker=self._ticker, position=0.0, confidence=0.0,
                regime=regime_name, regime_probs={}, volatility=0.2
            )
        
        current_vol = 0.20
        if 'rolling_volatility' in df.columns:
            v = df['rolling_volatility'].iloc[idx]
            if not pd.isna(v) and v > 0:
                current_vol = v
        
        current_date = df.index[idx]
        vix_val = self._get_value(self._vix_data, current_date)
        vix_tr = self._get_value(self._vix_term_ratio, current_date)
        wk_trend = self._get_value(self._weekly_trend, current_date)
        sec_trend = self._get_value(self._sector_trend, current_date)
        avg_corr = self._get_value(self._avg_correlation, current_date)
        
        gating = self._gating.compute_weights(regime_probs)
        
        agent_signals = {}
        for name in gating.active_agents:
            if name in self._agents:
                sig = self._agents[name].generate_signal(current_data, regime_probs)
                agent_signals[name] = sig
        
        if not agent_signals:
            return BaseModelSignal(
                ticker=self._ticker, position=0.0, confidence=0.0,
                regime=regime_name, regime_probs=regime_probs, volatility=current_vol
            )
        
        ensemble_out = self._ensemble.combine_signals(
            agent_signals, gating.weights, current_volatility=current_vol
        )
        
        ctrl = self._controller.compute_position(
            ensemble_out.final_action, ensemble_out.confidence,
            current_vol, regime_probs,
            vix_value=vix_val, vix_term_ratio=vix_tr,
            weekly_trend=wk_trend, sector_trend=sec_trend,
            avg_correlation=avg_corr,
            external_drawdown=portfolio_drawdown,
        )
        
        # Update controller state
        if ctrl.is_trade_allowed:
            prev_pos = self._controller.current_position
            self._controller.execute_trade(ctrl.target_position)
            if 'returns' in df.columns and idx > 0:
                daily_ret = df['returns'].iloc[idx]
                if not pd.isna(daily_ret):
                    self._controller.update_equity(prev_pos * daily_ret)
        
        return BaseModelSignal(
            ticker=self._ticker,
            position=ctrl.target_position,
            confidence=ensemble_out.confidence,
            regime=regime_name,
            regime_probs=regime_probs,
            volatility=current_vol,
            metadata={
                'baseline': ctrl.baseline,
                'overlay': ctrl.overlay,
                'drawdown': ctrl.metadata.get('drawdown', 0),
                'vix': vix_val,
                'weekly_trend': wk_trend,
            }
        )
    
    def reset(self) -> None:
        if self._controller:
            self._controller.reset()
        if self._ensemble:
            self._ensemble._action_history.clear()
