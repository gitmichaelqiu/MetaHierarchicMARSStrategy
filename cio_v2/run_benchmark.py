"""
CIO Benchmark Runner
Runs CIO portfolio backtests with different allocators.

Usage:
    conda run -n ml python cio/run_benchmark.py                     # Default: v1 allocator
    conda run -n ml python cio/run_benchmark.py --allocator v2      # Signal-weighted
    conda run -n ml python cio/run_benchmark.py --allocator all     # Compare all
"""

import sys
import os
import itertools
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIO_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')

sys.path.insert(0, CIO_DIR)
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from curl_cffi import requests as cffi_requests

from data_loader import DataLoader, get_benchmark_tickers
from indicators import add_all_indicators
from cio_orchestrator import CIOOrchestrator, CIOResult
from risk_manager import RiskManager
from allocator.v1_risk_parity import RiskParityAllocator
from allocator.v2_signal_weighted import SignalWeightedAllocator
from allocator.v3_mean_variance import MeanVarianceAllocator
from allocator.v4_conviction_weighted import ConvictionWeightedAllocator
from allocator.v5_absolute_conviction import AbsoluteConvictionAllocator
from adapters.v7_adapter import V7Adapter
from visualization import plot_all_visualizations


# ──── Tickers ────

CIO_TICKERS = [
    'AAPL', 'MSFT', 'GOOG', 'NVDA', 'TSLA',
    'AMD', 'TSM', 'AMZN', 'META', 'AVGO',
]


# ──── Cross-Asset Data ────

_SESSION = None

def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = cffi_requests.Session(impersonate="chrome")
    return _SESSION


def fetch_vix(period='2y'):
    try:
        vix = yf.Ticker('^VIX', session=_get_session())
        df = vix.history(period=period)
        df.columns = [c.lower() for c in df.columns]
        return df['close'].rename('vix')
    except Exception as e:
        print(f"  WARNING: VIX unavailable: {e}")
        return pd.Series(dtype=float, name='vix')


def fetch_vix3m(period='2y'):
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


def compute_vix_term_ratio(vix_data, vix3m_data):
    if vix_data.empty or vix3m_data.empty:
        return pd.Series(dtype=float)
    combined = pd.DataFrame({'vix': vix_data, 'vix3m': vix3m_data}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    return (combined['vix'] / combined['vix3m']).rename('vix_term_ratio')


def compute_rolling_correlation(all_returns, window=20):
    if len(all_returns) < 2:
        return pd.Series(dtype=float)
    ret_df = pd.DataFrame(all_returns).dropna()
    if len(ret_df) < window:
        return pd.Series(dtype=float)
    pairs = list(itertools.combinations(ret_df.columns, 2))
    pair_corrs = [ret_df[a].rolling(window).corr(ret_df[b]) for a, b in pairs]
    return pd.concat(pair_corrs, axis=1).mean(axis=1)


# ──── Plotting ────

def plot_cio_results(result: CIOResult, allocator_name: str, tickers, save_dir='../Plots'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'CIO Portfolio — {allocator_name}', fontsize=16, fontweight='bold')
    
    # Equity curve
    ax = axes[0]
    ax.plot(result.equity_curve.index, result.equity_curve.values,
            label=f'CIO ({result.total_return:.1%})', linewidth=2, color='#2196F3')
    ax.axhline(y=1 + result.equal_weight_return, color='#FF9800', linestyle='--',
               label=f'Equal-Weight B&H ({result.equal_weight_return:.1%})')
    ax.set_ylabel('Portfolio Value (normalized)')
    ax.set_title('Portfolio Equity Curve')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Drawdown
    ax = axes[1]
    ax.fill_between(result.drawdown_curve.index, 0, result.drawdown_curve.values,
                     color='#F44336', alpha=0.4)
    ax.set_ylabel('Drawdown')
    ax.set_title(f'Portfolio Drawdown (Max: {result.max_drawdown:.1%})')
    ax.grid(True, alpha=0.3)
    
    # Weight allocation
    ax = axes[2]
    wt_df = result.weight_history
    if not wt_df.empty:
        colors = plt.cm.Set3(np.linspace(0, 1, len(wt_df.columns)))
        ax.stackplot(wt_df.index, *[wt_df[c] for c in wt_df.columns],
                      labels=wt_df.columns, colors=colors, alpha=0.8)
        # Add cash
        cash = result.cash_history
        if not cash.empty:
            ax.fill_between(cash.index, sum(wt_df[c] for c in wt_df.columns),
                           1.0, color='#BDBDBD', alpha=0.5, label='Cash')
        ax.set_ylabel('Allocation')
        ax.set_title('Portfolio Weight Allocation')
        ax.legend(loc='upper right', ncol=4, fontsize=8)
        ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cio_{allocator_name.lower().replace(" ", "_")}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(results_by_allocator, save_dir='../Plots'):
    """Compare multiple allocators side by side."""
    os.makedirs(save_dir, exist_ok=True)
    if len(results_by_allocator) < 2:
        return
    
    names = list(results_by_allocator.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CIO Allocator Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(names))
    
    ax = axes[0]
    rets = [results_by_allocator[n].total_return * 100 for n in names]
    ax.bar(x, rets, color='#2196F3')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('Return (%)'); ax.set_title('Total Return')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    sharpes = [results_by_allocator[n].sharpe_ratio for n in names]
    ax.bar(x, sharpes, color='#4CAF50')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('Sharpe'); ax.set_title('Sharpe Ratio')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[2]
    dds = [results_by_allocator[n].max_drawdown * 100 for n in names]
    ax.bar(x, dds, color='#F44336')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('MaxDD (%)'); ax.set_title('Max Drawdown')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cio_allocator_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ──── Main ────

def run_cio_benchmark(allocator_choice='v1', tickers=None, period='2y'):
    if tickers is None:
        tickers = CIO_TICKERS
    
    print(f"\n{'='*70}")
    print(f"  CIO FRAMEWORK — Portfolio-Level Allocation Benchmark")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'='*70}")
    
    # 1. Fetch cross-asset data
    print("\n  Fetching cross-asset data...")
    vix_data = fetch_vix(period)
    vix3m_data = fetch_vix3m(period)
    vix_tr = compute_vix_term_ratio(vix_data, vix3m_data)
    
    xlk_weekly = fetch_weekly_data('XLK', period)
    sector_trend = compute_weekly_ema_trend(xlk_weekly)
    
    if not vix_data.empty:
        print(f"    VIX: {len(vix_data)} bars, latest={vix_data.iloc[-1]:.1f}")
    if not sector_trend.empty:
        print(f"    XLK trend: latest={sector_trend.iloc[-1]:+.3f}")
    
    # 2. Fetch ticker data
    print("  Fetching ticker data...")
    loader = DataLoader()
    ticker_data = {}
    all_returns = {}
    
    for t in tickers:
        try:
            df = loader.fetch_ticker(t, period=period)
            df = add_all_indicators(df)
            ticker_data[t] = df
            all_returns[t] = df['returns']
            print(f"    {t}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"    {t}: FAILED ({e})")
    
    # Remove failed tickers
    tickers = [t for t in tickers if t in ticker_data]
    
    avg_corr = compute_rolling_correlation(all_returns)
    if not avg_corr.empty:
        print(f"    Cross-correlation: mean={avg_corr.mean():.2f}, max={avg_corr.max():.2f}")
    
    # 3. Initialize adapters
    print("  Initializing V7 adapters...")
    adapters = {}
    for t in tickers:
        adapter = V7Adapter()
        
        wk_df = fetch_weekly_data(t, period)
        wk_trend = compute_weekly_ema_trend(wk_df) if not wk_df.empty else None
        
        adapter.initialize(
            t, ticker_data[t],
            vix_data=vix_data,
            vix_term_ratio=vix_tr if not vix_tr.empty else None,
            weekly_trend=wk_trend,
            sector_trend=sector_trend if not sector_trend.empty else None,
            avg_correlation=avg_corr if not avg_corr.empty else None,
        )
        adapters[t] = adapter
        print(f"    {t}: V7 adapter initialized")
    
    # 4. Run with selected allocator(s)
    allocators = {}
    if allocator_choice == 'all':
        allocators['V1 Risk Parity'] = RiskParityAllocator()
        allocators['V2 Signal-Weighted'] = SignalWeightedAllocator()
        allocators['V3 Mean-Variance'] = MeanVarianceAllocator()
        allocators['V4 Conviction-Weights'] = ConvictionWeightedAllocator()
        allocators['V5 Absolute Conviction'] = AbsoluteConvictionAllocator()
    elif allocator_choice == 'v2':
        allocators['V2 Signal-Weighted'] = SignalWeightedAllocator()
    elif allocator_choice == 'v3':
        allocators['V3 Mean-Variance'] = MeanVarianceAllocator()
    elif allocator_choice == 'v4':
        allocators['V4 Conviction-Weighted'] = ConvictionWeightedAllocator()
    elif allocator_choice == 'v5':
        allocators['V5 Absolute Conviction'] = AbsoluteConvictionAllocator()
    else:
        allocators['V1 Risk Parity'] = RiskParityAllocator()
    
    all_results = {}
    
    for alloc_name, allocator in allocators.items():
        print(f"\n{'='*70}")
        print(f"  Running CIO with {alloc_name}")
        print(f"{'='*70}")
        
        # Reset adapters for each run
        for adapter in adapters.values():
            adapter.reset()
        
        risk_mgr = RiskManager()
        cio = CIOOrchestrator(
            allocator=allocator,
            risk_manager=risk_mgr,
            debug=True,
        )
        
        result = cio.run(
            ticker_data=ticker_data,
            adapters=adapters,
            vix_data=vix_data if not vix_data.empty else None,
            avg_correlation=avg_corr if not avg_corr.empty else None,
        )
        
        print(f"\n  ┌──────────────────────────────────────────┐")
        print(f"  │  {alloc_name:^38}  │")
        print(f"  ├──────────────────────────────────────────┤")
        print(f"  │  Total Return:      {result.total_return:>10.2%}           │")
        print(f"  │  EW B&H Benchmark:  {result.equal_weight_return:>10.2%}           │")
        print(f"  │  Sharpe Ratio:      {result.sharpe_ratio:>10.2f}           │")
        print(f"  │  Sortino Ratio:     {result.sortino_ratio:>10.2f}           │")
        print(f"  │  Max Drawdown:      {result.max_drawdown:>10.2%}           │")
        print(f"  │  Avg Cash %:        {result.avg_cash_pct:>10.1%}           │")
        print(f"  │  Rebalances:        {result.num_rebalances:>10d}           │")
        print(f"  └──────────────────────────────────────────┘")
        
        print(f"\n  Per-Ticker Attribution:")
        print(f"  {'Ticker':>6}  {'Alloc Contrib':>14}  {'B&H Return':>12}")
        for t in tickers:
            contrib = result.ticker_contributions.get(t, 0)
            bh = result.ticker_returns.get(t, 0)
            print(f"  {t:>6}  {contrib:>+14.2%}  {bh:>12.2%}")
        
        plot_cio_results(result, alloc_name, tickers)
        plot_all_visualizations(result, ticker_data, tickers,
                                 allocator_name=alloc_name)
        all_results[alloc_name] = result
    
    # Compare if multiple
    if len(all_results) > 1:
        plot_comparison(all_results)
        print(f"\n  {'='*70}")
        print(f"  ALLOCATOR COMPARISON")
        print(f"  {'='*70}")
        print(f"  {'Allocator':>25}  {'Return':>10}  {'Sharpe':>8}  {'MaxDD':>8}  {'Cash':>8}")
        for name, r in all_results.items():
            print(f"  {name:>25}  {r.total_return:>10.2%}  {r.sharpe_ratio:>8.2f}  "
                  f"{r.max_drawdown:>8.2%}  {r.avg_cash_pct:>8.1%}")
    
    print(f"\n  Plots saved to Plots/")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIO Portfolio Benchmark')
    parser.add_argument('--allocator', choices=['v1', 'v2', 'v3', 'v4', 'v5', 'all'], default='all',
                        help='Allocator to use (default: all)')
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Override ticker list')
    args = parser.parse_args()
    
    run_cio_benchmark(
        allocator_choice=args.allocator,
        tickers=args.tickers,
    )
