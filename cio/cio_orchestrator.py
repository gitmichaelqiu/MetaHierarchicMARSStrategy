"""
CIO Orchestrator — The main portfolio management engine.

Runs the daily loop:
  1. Get base model signals for all tickers
  2. Assess portfolio risk
  3. Compute allocation weights (via allocator)
  4. Apply risk constraints
  5. Execute rebalancing if needed
  6. Track portfolio equity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field

from base_model_adapter import BaseModelAdapter, BaseModelSignal
from allocator.base import BaseAllocator, AllocationOutput
from risk_manager import RiskManager, RiskOutput


@dataclass
class PortfolioState:
    """Current portfolio state at a point in time."""
    date: pd.Timestamp
    weights: Dict[str, float]         # ticker -> allocation weight
    positions: Dict[str, float]       # ticker -> base model position signal
    cash_pct: float
    equity: float
    risk_flags: List[str]
    day_return: float = 0.0


@dataclass
class CIOResult:
    """Full CIO backtest result."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_cash_pct: float
    num_rebalances: int
    
    # Curves
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdown_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    weight_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    cash_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    
    # Per-ticker
    ticker_returns: Dict[str, float] = field(default_factory=dict)
    ticker_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Comparison
    equal_weight_return: float = 0.0
    bh_return: float = 0.0


class CIOOrchestrator:
    """
    Chief Investment Officer — portfolio-level allocation engine.
    """
    
    def __init__(
        self,
        allocator: BaseAllocator,
        risk_manager: Optional[RiskManager] = None,
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        rebalance_threshold: float = 0.05,
        risk_free_rate: float = 0.04,
        debug: bool = True,
    ):
        self.allocator = allocator
        self.risk_manager = risk_manager or RiskManager()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_threshold = rebalance_threshold
        self.risk_free_rate = risk_free_rate
        self.debug = debug
    
    def run(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        adapters: Dict[str, BaseModelAdapter],
        vix_data: Optional[pd.Series] = None,
        avg_correlation: Optional[pd.Series] = None,
        warmup: int = 60,
    ) -> CIOResult:
        """
        Run portfolio backtest.
        
        Args:
            ticker_data: {ticker: DataFrame} with OHLCV + indicators
            adapters: {ticker: initialized BaseModelAdapter}
            vix_data: VIX daily close series
            avg_correlation: Rolling avg correlation series
            warmup: Warmup bars to skip
        """
        tickers = list(ticker_data.keys())
        
        # Find common date range
        all_dates = None
        for t, df in ticker_data.items():
            idx = df.index[warmup:]
            if all_dates is None:
                all_dates = set(idx)
            else:
                all_dates &= set(idx)
        
        if not all_dates:
            raise ValueError("No overlapping dates across tickers")
        
        dates = sorted(all_dates)
        n_days = len(dates)
        
        if self.debug:
            print(f"  Portfolio: {len(tickers)} tickers, {n_days} overlapping bars")
            print(f"  Date range: {dates[0]} to {dates[-1]}")
            print(f"  Allocator: {self.allocator.version}")
        
        # State tracking
        equity = self.initial_capital
        peak_equity = equity
        current_weights: Dict[str, float] = {t: 0.0 for t in tickers}
        current_cash_pct = 1.0
        
        equity_history = []
        weight_history = []
        cash_history = []
        drawdown_history = []
        daily_returns = []
        ticker_cum_contrib = {t: 0.0 for t in tickers}
        num_rebalances = 0
        
        # Pre-compute index mappings for each ticker
        ticker_idx = {}
        for t, df in ticker_data.items():
            idx_map = {}
            for i, d in enumerate(df.index):
                d_norm = pd.Timestamp(d)
                if hasattr(d_norm, 'tz') and d_norm.tz is not None:
                    d_norm = d_norm.tz_convert(None)
                d_norm = d_norm.normalize()
                idx_map[d_norm] = i
            ticker_idx[t] = idx_map
        
        for day_i, date in enumerate(dates):
            date_norm = pd.Timestamp(date)
            if hasattr(date_norm, 'tz') and date_norm.tz is not None:
                date_norm = date_norm.tz_convert(None)
            date_norm = date_norm.normalize()
            
            # 1. Get all base model signals
            signals: Dict[str, BaseModelSignal] = {}
            daily_rets: Dict[str, float] = {}
            
            for t in tickers:
                idx = ticker_idx[t].get(date_norm)
                if idx is None or idx < warmup:
                    continue
                
                df = ticker_data[t]
                sig = adapters[t].get_signal(df, idx)
                signals[t] = sig
                
                # Daily return for this ticker
                if idx > 0 and 'returns' in df.columns:
                    r = df['returns'].iloc[idx]
                    daily_rets[t] = r if not pd.isna(r) else 0.0
                else:
                    daily_rets[t] = 0.0
            
            if not signals:
                equity_history.append(equity)
                weight_history.append({t: 0.0 for t in tickers})
                cash_history.append(1.0)
                drawdown_history.append(0.0)
                daily_returns.append(0.0)
                continue
            
            # 2. Get VIX and correlation for today
            vix_val = self._get_value(vix_data, date_norm)
            corr_val = self._get_value(avg_correlation, date_norm)
            
            # 3. Assess portfolio risk
            risk = self.risk_manager.assess_risk(
                signals, vix_value=vix_val, avg_correlation=corr_val
            )
            
            # 4. Compute allocation
            allocation = self.allocator.allocate(signals, max_exposure=risk.max_exposure)
            
            # Apply risk constraints and base model conviction
            target_weights = {}
            for t, alloc_w in allocation.weights.items():
                if t in signals:
                    base_pos = signals[t].position
                    if base_pos <= 0.05:
                        # Model says near-zero — don't allocate
                        target_weights[t] = 0.0
                    elif base_pos < 0.3:
                        # Weak conviction — scale down proportionally
                        target_weights[t] = alloc_w * (base_pos / 0.3) * risk.position_scalar
                    else:
                        # Strong conviction — use full allocation weight
                        target_weights[t] = alloc_w * risk.position_scalar
                else:
                    target_weights[t] = 0.0
            
            # Fill missing tickers
            for t in tickers:
                if t not in target_weights:
                    target_weights[t] = 0.0
            
            # 5. Check if rebalance needed
            should_rebalance = day_i == 0  # Always on first day
            if not should_rebalance:
                max_drift = max(
                    abs(target_weights.get(t, 0) - current_weights.get(t, 0))
                    for t in tickers
                )
                should_rebalance = max_drift > self.rebalance_threshold
                
                # Also rebalance on regime change
                if not should_rebalance:
                    for t, sig in signals.items():
                        if sig.regime == 'Crisis' and current_weights.get(t, 0) > 0.1:
                            should_rebalance = True
                            break
            
            # 6. Execute rebalance
            if should_rebalance:
                # Transaction costs
                turnover = sum(
                    abs(target_weights.get(t, 0) - current_weights.get(t, 0))
                    for t in tickers
                )
                cost = turnover * self.transaction_cost * equity
                equity -= cost
                current_weights = target_weights.copy()
                current_cash_pct = 1.0 - sum(current_weights.values())
                num_rebalances += 1
            
            # 7. Compute portfolio return
            port_ret = 0.0
            for t in tickers:
                w = current_weights.get(t, 0.0)
                r = daily_rets.get(t, 0.0)
                contrib = w * r
                port_ret += contrib
                ticker_cum_contrib[t] += contrib
            
            # Feed returns to v3 allocator for covariance estimation
            if hasattr(self.allocator, 'update_returns'):
                self.allocator.update_returns(daily_rets)
            
            # Cash return (daily risk-free)
            port_ret += current_cash_pct * (self.risk_free_rate / 252)
            
            equity *= (1 + port_ret)
            peak_equity = max(peak_equity, equity)
            dd = max(0, 1 - equity / peak_equity)
            
            # Update risk manager equity
            self.risk_manager.update_equity(port_ret)
            
            # Record
            equity_history.append(equity)
            weight_history.append(current_weights.copy())
            cash_history.append(current_cash_pct)
            drawdown_history.append(dd)
            daily_returns.append(port_ret)
            
            # Debug output
            if self.debug and day_i % 50 == 0:
                w_str = ", ".join(f"{t}:{current_weights.get(t,0):.0%}" for t in tickers[:5])
                flag_str = ", ".join(risk.risk_flags) if risk.risk_flags else "OK"
                print(f"    [{str(date)[:10]}] Cash: {current_cash_pct:.0%}, "
                      f"Weights: [{w_str}], "
                      f"Eq: {equity/self.initial_capital:.3f}, DD: {dd:.1%}, "
                      f"Risk: {flag_str}")
        
        # Compute metrics
        ret_series = pd.Series(daily_returns, index=dates)
        eq_series = pd.Series(equity_history, index=dates) / self.initial_capital
        dd_series = pd.Series(drawdown_history, index=dates)
        cash_series = pd.Series(cash_history, index=dates)
        wt_df = pd.DataFrame(weight_history, index=dates)
        
        total_ret = equity / self.initial_capital - 1
        n_years = n_days / 252
        ann_ret = (1 + total_ret) ** (1 / max(0.01, n_years)) - 1
        
        vol = ret_series.std() * np.sqrt(252)
        excess = ret_series.mean() * 252 - self.risk_free_rate
        sharpe = excess / vol if vol > 0 else 0
        
        downside = ret_series[ret_series < 0].std() * np.sqrt(252)
        sortino = excess / downside if downside > 0 else 0
        
        max_dd = dd_series.max()
        
        # Equal-weight B&H benchmark
        ew_ret = 0.0
        for t in tickers:
            df = ticker_data[t]
            if len(df) > warmup:
                start_p = df['close'].iloc[warmup]
                end_p = df['close'].iloc[-1]
                ew_ret += (end_p / start_p - 1)
        ew_ret /= len(tickers)
        
        # Per-ticker returns
        ticker_rets = {}
        for t in tickers:
            df = ticker_data[t]
            if len(df) > warmup:
                start_p = df['close'].iloc[warmup]
                end_p = df['close'].iloc[-1]
                ticker_rets[t] = end_p / start_p - 1
        
        return CIOResult(
            total_return=total_ret,
            annualized_return=ann_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            avg_cash_pct=cash_series.mean(),
            num_rebalances=num_rebalances,
            equity_curve=eq_series,
            drawdown_curve=dd_series,
            weight_history=wt_df,
            cash_history=cash_series,
            ticker_returns=ticker_rets,
            ticker_contributions=ticker_cum_contrib,
            equal_weight_return=ew_ret,
            bh_return=ew_ret,
        )
    
    def _get_value(self, series, date):
        if series is None or (isinstance(series, pd.Series) and series.empty):
            return None
        try:
            idx = series.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx = idx.tz_convert(None)
            mask = idx.normalize() <= date
            if mask.any():
                return float(series.iloc[idx[mask].get_loc(idx[mask][-1])])
        except Exception:
            pass
        return None
