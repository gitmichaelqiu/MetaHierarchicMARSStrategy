import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
V7_DIR = os.path.join(PROJECT_ROOT, 'v7_improved')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, V7_DIR)

from data_loader import DataLoader
from indicators import add_all_indicators
from backtest_engine import BacktestEngine
from v7_improved.run_benchmark import (
    MoAStrategyV7, fetch_vix_data, fetch_vix3m_data, 
    compute_vix_term_ratio, fetch_weekly_data, compute_weekly_ema_trend,
    compute_rolling_correlation, plot_results
)

def run_stress_test(ticker, start_date_str, end_date_str):
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {ticker} from {start_date_str} to {end_date_str}")
    print(f"{'='*60}")
    
    # We need historical data for indicators (warmup)
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    fetch_start = (start_date - timedelta(days=150)).strftime('%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    loader = DataLoader()
    
    # Fetch ticker data
    df = loader.fetch_ticker(ticker, start_date=fetch_start, end_date=end_date_str)
    df = add_all_indicators(df)
    
    # Find the warmup_period such that trading starts at start_date
    # Find the index of the first date >= start_date
    start_idx = 0
    for i, date in enumerate(df.index):
        if date.tz_localize(None) >= start_date:
            start_idx = i
            break
    
    # Ensure we have at least 60 bars for initial regime detection/indicators
    if start_idx < 60:
        print(f"  WARNING: Only {start_idx} bars available for warmup. Model may be unstable.")
    
    # Fetch cross-asset data for the same period
    vix_data = fetch_vix_data(period='5y') # Fetch more and filter
    vix3m_data = fetch_vix3m_data(period='5y')
    vix_term_ratio = compute_vix_term_ratio(vix_data, vix3m_data)
    
    xlk_weekly = fetch_weekly_data('XLK', period='5y')
    sector_trend = compute_weekly_ema_trend(xlk_weekly)
    
    ticker_weekly = fetch_weekly_data(ticker, period='5y')
    wk_trend = compute_weekly_ema_trend(ticker_weekly)
    
    # Correlation (simplified for single ticker stress test vs benchmark pool)
    # We'll just use the ticker's own volatility as a proxy if we don't have the full pool
    # But for V7, avg_correlation is expected. We'll pass None and see if it handles it.
    
    strategy = MoAStrategyV7(
        vix_data=vix_data,
        vix_term_ratio=vix_term_ratio,
        weekly_trend=wk_trend,
        sector_trend=sector_trend,
        avg_correlation=None, # In a single ticker test, we don't have "average cross-correlation"
        debug=False
    )
    strategy.regime_detector.fit(df.iloc[:start_idx])
    
    engine = BacktestEngine(transaction_cost=0.001)
    result = engine.run(df, strategy.generate_signal, warmup_period=start_idx)
    
    print(f"  Results for {ticker} ({start_date_str} to {end_date_str}):")
    print(f"    Total Return:   {result.total_return:>8.2%}")
    print(f"    Benchmark:      {result.benchmark_return:>8.2%}")
    print(f"    Alpha:          {result.alpha:>8.2%}")
    print(f"    Sharpe:         {result.sharpe_ratio:>8.2f}")
    print(f"    Max Drawdown:   {result.max_drawdown:>8.2%}")
    
    # Save plots
    save_dir = os.path.join(PROJECT_ROOT, 'StressTestPlots')
    os.makedirs(save_dir, exist_ok=True)
    
    # Custom plotting to include dates in filename
    plot_results(f"{ticker}_{start_date_str}_{end_date_str}", result, save_dir=save_dir)
    
    return result

if __name__ == "__main__":
    tests = [
        ('TSLA', '2024-12-16', '2025-04-22'),
        ('MSFT', '2025-10-29', '2026-02-12'),
        ('AMZN', '2025-02-06', '2025-04-22'),
    ]
    
    summaries = []
    for ticker, start, end in tests:
        try:
            res = run_stress_test(ticker, start, end)
            summaries.append({
                'Ticker': ticker,
                'Start': start,
                'End': end,
                'Return': res.total_return,
                'Benchmark': res.benchmark_return,
                'Alpha': res.alpha,
                'Sharpe': res.sharpe_ratio,
                'MaxDD': res.max_drawdown
            })
        except Exception as e:
            print(f"FAILED {ticker} stress test: {e}")
            import traceback
            traceback.print_exc()

    if summaries:
        print(f"\n{'='*80}")
        print(f"{'Ticker':<6} {'Start':<12} {'End':<12} {'Return':>8} {'Bench':>8} {'Alpha':>8} {'Sharpe':>8} {'MaxDD':>8}")
        print("-" * 80)
        for s in summaries:
            print(f"{s['Ticker']:<6} {s['Start']:<12} {s['End']:<12} {s['Return']:>8.1%} {s['Benchmark']:>8.1%} {s['Alpha']:>8.1%} {s['Sharpe']:>8.2f} {s['MaxDD']:>8.1%}")
        print(f"{'='*80}")
