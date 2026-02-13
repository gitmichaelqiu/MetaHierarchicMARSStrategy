"""
V7 MoA Trading Framework — Benchmark Runner
Cross-Asset Intelligence: Sector Momentum + Correlation Risk + VIX Term Structure.

Usage:
    conda run -n ml python run_benchmark.py
"""

import sys
import os
import itertools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V7_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, V7_DIR)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from curl_cffi import requests as cffi_requests

from data_loader import DataLoader, get_benchmark_tickers
from indicators import add_all_indicators
from backtest_engine import BacktestEngine, BacktestResult

from regime_detector import RegimeDetector
from moa_gating import MoAGatingNetwork
from moa_ensemble import MoASoftEnsemble
from meta_controller import MetaController, ControllerOutput

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.volatility_agent import VolatilityAgent
from agents.crisis_agent import CrisisAgent
from v7_agents.exponential_momentum_agent import ExponentialMomentumAgent


# ──────────────────────────────────────────
# Cross-Asset Data Fetchers
# ──────────────────────────────────────────

# Shared session to avoid rate-limit issues from creating multiple sessions
_SHARED_SESSION = None

def _get_session():
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        _SHARED_SESSION = cffi_requests.Session(impersonate="chrome")
    return _SHARED_SESSION


def fetch_vix_data(period='2y'):
    try:
        vix = yf.Ticker('^VIX', session=_get_session())
        df = vix.history(period=period)
        df.columns = [c.lower() for c in df.columns]
        return df['close'].rename('vix')
    except Exception as e:
        print(f"  WARNING: VIX unavailable: {e}")
        return pd.Series(dtype=float, name='vix')


def fetch_vix3m_data(period='2y'):
    try:
        vix3m = yf.Ticker('^VIX3M', session=_get_session())
        df = vix3m.history(period=period)
        df.columns = [c.lower() for c in df.columns]
        return df['close'].rename('vix3m')
    except Exception as e:
        print(f"  WARNING: VIX3M unavailable: {e}")
        return pd.Series(dtype=float, name='vix3m')


def fetch_weekly_data(ticker, period='2y'):
    try:
        t = yf.Ticker(ticker, session=_get_session())
        df = t.history(period=period, interval='1wk')
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"  WARNING: Weekly data unavailable for {ticker}: {e}")
        return pd.DataFrame()


def compute_weekly_ema_trend(weekly_df, fast=13, slow=26):
    if weekly_df.empty or len(weekly_df) < slow:
        return pd.Series(dtype=float)
    close = weekly_df['close']
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return (ema_f - ema_s) / ema_s


def compute_rolling_correlation(all_daily_returns: Dict[str, pd.Series], window=20) -> pd.Series:
    """
    Compute average pairwise rolling correlation across all tickers.
    Returns a daily series of average correlation values.
    """
    if len(all_daily_returns) < 2:
        return pd.Series(dtype=float)
    
    # Align all return series
    ret_df = pd.DataFrame(all_daily_returns)
    ret_df = ret_df.dropna()
    
    if len(ret_df) < window:
        return pd.Series(dtype=float)
    
    tickers = list(ret_df.columns)
    pairs = list(itertools.combinations(tickers, 2))
    
    pair_corrs = []
    for a, b in pairs:
        corr = ret_df[a].rolling(window).corr(ret_df[b])
        pair_corrs.append(corr)
    
    avg_corr = pd.concat(pair_corrs, axis=1).mean(axis=1)
    return avg_corr


def compute_vix_term_ratio(vix_data, vix3m_data):
    """Compute VIX / VIX3M ratio. >1.0 = inverted (near-term stress)."""
    if vix_data.empty or vix3m_data.empty:
        return pd.Series(dtype=float)
    
    combined = pd.DataFrame({'vix': vix_data, 'vix3m': vix3m_data}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    
    return (combined['vix'] / combined['vix3m']).rename('vix_term_ratio')


# ──────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────

class MoAStrategyV7:
    """V7 MoA: Cross-Asset Intelligence."""
    
    def __init__(
        self,
        vix_data=None,
        vix_term_ratio=None,
        weekly_trend=None,
        sector_trend=None,
        avg_correlation=None,
        debug=True
    ):
        self.regime_detector = RegimeDetector()
        self.agents = {
            'TrendAgent': TrendAgent(),
            'MeanReversionAgent': MeanReversionAgent(),
            'VolatilityAgent': VolatilityAgent(),
            'CrisisAgent': CrisisAgent(),
            'ExponentialMomentumAgent': ExponentialMomentumAgent(),
        }
        self.gating = MoAGatingNetwork(top_k=3, temperature=0.8)
        self.ensemble = MoASoftEnsemble()
        self.controller = MetaController()
        
        self.vix_data = vix_data
        self.vix_term_ratio = vix_term_ratio
        self.weekly_trend = weekly_trend
        self.sector_trend = sector_trend
        self.avg_correlation = avg_correlation
        self.debug = debug
        self._agent_activation_count = {n: 0 for n in self.agents}
        self._agent_contribution_sum = {n: 0.0 for n in self.agents}
    
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
    
    def generate_signal(self, df, current_idx):
        current_data = df.iloc[:current_idx + 1]
        if len(current_data) < 60:
            return 0.0, 'Unknown'
        
        regime_name, regime_probs = self.regime_detector.get_current_regime(current_data)
        if not regime_probs:
            return 0.0, regime_name
        
        current_vol = 0.20
        if 'rolling_volatility' in df.columns:
            v = df['rolling_volatility'].iloc[current_idx]
            if not pd.isna(v) and v > 0:
                current_vol = v
        
        current_date = df.index[current_idx]
        vix_val = self._get_value(self.vix_data, current_date)
        vix_tr = self._get_value(self.vix_term_ratio, current_date)
        wk_trend = self._get_value(self.weekly_trend, current_date)
        sec_trend = self._get_value(self.sector_trend, current_date)
        avg_corr = self._get_value(self.avg_correlation, current_date)
        
        gating_output = self.gating.compute_weights(regime_probs)
        
        agent_signals = {}
        for name in gating_output.active_agents:
            if name in self.agents:
                signal = self.agents[name].generate_signal(current_data, regime_probs)
                agent_signals[name] = signal
                self._agent_activation_count[name] += 1
                self._agent_contribution_sum[name] += abs(signal.action * signal.confidence)
        
        if not agent_signals:
            return 0.0, regime_name
        
        ensemble_out = self.ensemble.combine_signals(
            agent_signals, gating_output.weights, current_volatility=current_vol
        )
        
        ctrl = self.controller.compute_position(
            ensemble_out.final_action, ensemble_out.confidence,
            current_vol, regime_probs,
            vix_value=vix_val, vix_term_ratio=vix_tr,
            weekly_trend=wk_trend, sector_trend=sec_trend,
            avg_correlation=avg_corr,
        )
        
        if ctrl.is_trade_allowed:
            prev_pos = self.controller.current_position
            self.controller.execute_trade(ctrl.target_position)
            if 'returns' in df.columns and current_idx > 0:
                daily_ret = df['returns'].iloc[current_idx]
                if not pd.isna(daily_ret):
                    self.controller.update_equity(prev_pos * daily_ret)
        
        if self.debug and current_idx % 50 == 0:
            d = str(df.index[current_idx])[:10]
            dd = ctrl.metadata.get('drawdown', 0)
            parts = [f"[{d}] {regime_name}, Pos: {ctrl.target_position:.2f}, Base: {ctrl.baseline:.2f}, Ovrl: {ctrl.overlay:+.2f}, DD: {dd:.1%}"]
            if vix_val: parts.append(f"VIX={vix_val:.1f}")
            if vix_tr: parts.append(f"TR={vix_tr:.2f}")
            if wk_trend is not None: parts.append(f"Wk={wk_trend:+.3f}")
            if sec_trend is not None: parts.append(f"Sec={sec_trend:+.3f}")
            if avg_corr is not None: parts.append(f"Corr={avg_corr:.2f}")
            print(f"    {', '.join(parts)}")
            for name, sig in agent_signals.items():
                w = gating_output.weights.get(name, 0)
                print(f"      {name}: act={sig.action:.3f}, conf={sig.confidence:.2f}, w={w:.2f}")
        
        return ctrl.target_position, regime_name
    
    def get_agent_summary(self):
        lines = ["  Per-Agent Summary:"]
        for n in self.agents:
            c = self._agent_activation_count[n]
            a = self._agent_contribution_sum[n] / max(1, c)
            lines.append(f"    {n}: active {c} bars, avg |act*conf|={a:.3f}")
        return '\n'.join(lines)
    
    def reset(self):
        self.controller.reset()
        self.ensemble._action_history.clear()
        self._agent_activation_count = {n: 0 for n in self.agents}
        self._agent_contribution_sum = {n: 0.0 for n in self.agents}


# ──────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────

def plot_results(ticker, result, save_dir='../Plots'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'V7 MoA Strategy: {ticker}', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    ax.plot(result.equity_curve.index, result.equity_curve.values,
            label=f'MoA V7 ({result.total_return:.1%})', linewidth=2, color='#2196F3')
    ax.plot(result.benchmark_curve.index, result.benchmark_curve.values,
            label=f'Buy & Hold ({result.benchmark_return:.1%})', linewidth=1.5,
            color='#FF9800', linestyle='--')
    ax.set_ylabel('Portfolio Value'); ax.legend(loc='upper left')
    ax.set_title('Equity Curve'); ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.fill_between(result.drawdown_curve.index, 0, result.drawdown_curve.values, color='#F44336', alpha=0.4)
    ax.set_ylabel('Drawdown'); ax.set_title(f'Drawdown (Max: {result.max_drawdown:.1%})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                     where=result.position_history >= 0, color='#4CAF50', alpha=0.5, label='Long')
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                     where=result.position_history < 0, color='#F44336', alpha=0.5, label='Short')
    ax.set_ylabel('Position'); ax.set_title('Position History')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3); ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax = axes[3]
    for rn, cl in {'Growth': '#4CAF50', 'Stagnation': '#9E9E9E', 'Transition': '#FF9800', 'Crisis': '#F44336', 'Unknown': '#BDBDBD'}.items():
        m = result.regime_history == rn
        if m.any(): ax.fill_between(result.regime_history.index, 0, 1, where=m, color=cl, alpha=0.4, label=rn)
    ax.set_ylabel('Regime'); ax.set_title('Regime History')
    ax.legend(loc='upper right', ncol=4); ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'v7_{ticker}_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(all_results, save_dir='../Plots'):
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(all_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('V7 MoA Framework — Cross-Asset Benchmark', fontsize=16, fontweight='bold')
    
    x = np.arange(len(tickers)); w = 0.35
    ax = axes[0]
    ax.bar(x-w/2, [all_results[t].total_return*100 for t in tickers], w, label='MoA V7', color='#2196F3')
    ax.bar(x+w/2, [all_results[t].benchmark_return*100 for t in tickers], w, label='Buy & Hold', color='#FF9800')
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel('Return (%)'); ax.set_title('Returns'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    al = [all_results[t].alpha*100 for t in tickers]
    ax.bar(tickers, al, color=['#4CAF50' if a>=0 else '#F44336' for a in al])
    ax.set_ylabel('Alpha (%)'); ax.set_title('Alpha'); ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax = axes[2]
    sh = [all_results[t].sharpe_ratio for t in tickers]
    ax.bar(tickers, sh, color=['#4CAF50' if s>=0 else '#F44336' for s in sh])
    ax.set_ylabel('Sharpe'); ax.set_title('Sharpe Ratio'); ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'v7_benchmark_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ──────────────────────────────────────────
# Main Benchmark
# ──────────────────────────────────────────

def run_full_benchmark(tickers=None, period='2y', debug=True):
    if tickers is None:
        tickers = get_benchmark_tickers()
    
    print(f"\n{'='*60}")
    print(f"MoA V7 — Cross-Asset Intelligence Benchmark")
    print(f"  Sector Momentum + Correlation Risk + VIX Term Structure")
    print(f"{'='*60}")
    
    # 1. Fetch cross-asset data
    print("  Fetching VIX...")
    vix_data = fetch_vix_data(period)
    if not vix_data.empty:
        print(f"    VIX: {len(vix_data)} bars, range [{vix_data.min():.1f}, {vix_data.max():.1f}]")
    
    print("  Fetching VIX3M...")
    vix3m_data = fetch_vix3m_data(period)
    vix_term_ratio = compute_vix_term_ratio(vix_data, vix3m_data)
    if not vix_term_ratio.empty:
        print(f"    VIX Term Ratio: range [{vix_term_ratio.min():.2f}, {vix_term_ratio.max():.2f}]")
    
    print("  Fetching XLK sector data...")
    xlk_weekly = fetch_weekly_data('XLK', period)
    sector_trend = compute_weekly_ema_trend(xlk_weekly)
    if not sector_trend.empty:
        print(f"    XLK trend: {len(sector_trend)} bars, latest={sector_trend.iloc[-1]:+.3f}")
    
    # 2. Fetch daily returns for all tickers (for correlation)
    print("  Computing cross-correlations...")
    loader = DataLoader()
    all_daily_returns = {}
    ticker_data = {}
    for ticker in tickers:
        try:
            df = loader.fetch_ticker(ticker, period=period)
            df = add_all_indicators(df)
            ticker_data[ticker] = df
            all_daily_returns[ticker] = df['returns']
            print(f"    {ticker}: {len(df)} bars")
        except Exception as e:
            print(f"    {ticker}: FAILED ({e})")
    
    avg_correlation = compute_rolling_correlation(all_daily_returns)
    if not avg_correlation.empty:
        print(f"    Avg corr: mean={avg_correlation.mean():.2f}, "
              f"high (>0.6) = {(avg_correlation > 0.6).mean():.0%}")
    
    # 3. Fetch weekly trend per ticker and run backtests
    all_results = {}
    for ticker in tickers:
        if ticker not in ticker_data:
            continue
        
        print(f"\n{'='*60}")
        print(f"Running V7 backtest for {ticker}")
        print(f"{'='*60}")
        
        # Per-ticker weekly trend
        wk_df = fetch_weekly_data(ticker, period)
        wk_trend = compute_weekly_ema_trend(wk_df) if not wk_df.empty else None
        
        df = ticker_data[ticker]
        strategy = MoAStrategyV7(
            vix_data=vix_data,
            vix_term_ratio=vix_term_ratio if not vix_term_ratio.empty else None,
            weekly_trend=wk_trend,
            sector_trend=sector_trend if not sector_trend.empty else None,
            avg_correlation=avg_correlation if not avg_correlation.empty else None,
            debug=debug,
        )
        strategy.regime_detector.fit(df)
        
        engine = BacktestEngine(transaction_cost=0.001)
        try:
            result = engine.run(df, strategy.generate_signal, warmup_period=60)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        
        print(f"\n  Results for {ticker}:")
        print(f"    Total Return:   {result.total_return:>8.2%}")
        print(f"    Benchmark:      {result.benchmark_return:>8.2%}")
        print(f"    Alpha:          {result.alpha:>8.2%}")
        print(f"    Sharpe:         {result.sharpe_ratio:>8.2f}")
        print(f"    Max Drawdown:   {result.max_drawdown:>8.2%}")
        print(f"    Num Trades:     {result.num_trades:>8d}")
        print(strategy.get_agent_summary())
        
        pos = result.position_history
        print(f"  Position Analysis:")
        print(f"    Avg position:       {pos.mean():>+.3f}")
        print(f"    Time near-zero:     {(pos.abs() < 0.1).mean():>.1%}")
        print(f"    Time long (>0.3):   {(pos > 0.3).mean():>.1%}")
        
        plot_results(ticker, result)
        all_results[ticker] = result
    
    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"V7 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Ticker':>6}  {'Return':>10}  {'B&H':>10}  {'Alpha':>8}  {'Sharpe':>8}  "
              f"{'MaxDD':>8}  {'Trades':>8}  {'AvgPos':>8}")
        
        for t, r in all_results.items():
            ap = r.position_history.mean()
            print(f"{t:>6}  {r.total_return:>10.2%}  {r.benchmark_return:>10.2%}  "
                  f"{r.alpha:>8.2%}  {r.sharpe_ratio:>8.2f}  "
                  f"{r.max_drawdown:>8.2%}  {r.num_trades:>8d}  {ap:>+8.3f}")
        
        ar = np.mean([r.total_return for r in all_results.values()])
        ab = np.mean([r.benchmark_return for r in all_results.values()])
        aa = np.mean([r.alpha for r in all_results.values()])
        ash = np.mean([r.sharpe_ratio for r in all_results.values()])
        ad = np.mean([r.max_drawdown for r in all_results.values()])
        
        print(f"\n  Avg Return:       {ar:.2%}")
        print(f"  Avg Benchmark:    {ab:.2%}")
        print(f"  Avg Alpha:        {aa:.2%}")
        print(f"  Avg Sharpe:       {ash:.2f}")
        print(f"  Avg MaxDD:        {ad:.2%}")
        print(f"  B&H Capture:      {ar/ab:.0%}")
        
        plot_summary(all_results)
        print(f"\n  Plots saved to Plots/")
    
    return all_results


if __name__ == '__main__':
    run_full_benchmark()
