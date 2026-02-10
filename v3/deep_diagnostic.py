"""
Deep diagnostic: WHY does V3 structurally underperform B&H?
Hypothesis: The baseline position is 0 (flat), so even with strong signals,
average position is <<1.0. B&H is always 1.0.
"""
import sys, os
V3_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V3_DIR)
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, V3_DIR)

import pandas as pd, numpy as np
from data_loader import DataLoader
from indicators import add_all_indicators
from run_benchmark import MoAStrategyV3
from backtest_engine import BacktestEngine

loader = DataLoader()

print("="*70)
print("V3 STRUCTURAL DIAGNOSTIC: Why does V3 miss buy-and-hold?")
print("="*70)

for ticker in ['AAPL', 'GOOG', 'NVDA', 'TSLA']:
    df = loader.fetch_ticker(ticker, period='2y')
    df = add_all_indicators(df)
    
    strategy = MoAStrategyV3(debug=False)
    strategy.regime_detector.fit(df)
    engine = BacktestEngine()
    result = engine.run(df, strategy.generate_signal, warmup_period=60)
    
    pos = result.position_history
    reg = result.regime_history
    
    print(f"\n{'='*70}")
    print(f"  {ticker}: V3={result.total_return:.1%} vs B&H={result.benchmark_return:.1%} (gap={result.total_return - result.benchmark_return:+.1%})")
    print(f"{'='*70}")
    
    # The fundamental question: what avg position would match B&H?
    # Answer: exactly 1.0 (always fully long)
    print(f"\n  POSITION DISTRIBUTION:")
    print(f"    Avg position:     {pos.mean():+.3f}  (B&H = +1.000)")
    print(f"    Avg |position|:   {pos.abs().mean():.3f}")
    print(f"    Long (>0.1):      {(pos > 0.1).mean():.0%}")
    print(f"    Short (<-0.1):    {(pos < -0.1).mean():.0%}")
    print(f"    Near-zero:        {(pos.abs() < 0.1).mean():.0%}")
    
    # Quantify: how much return is lost to each position bucket?
    daily_ret = df['returns'].reindex(pos.index).fillna(0)
    
    # Split returns by position state
    long_contrib = (daily_ret * pos * (pos > 0.1)).sum()
    short_contrib = (daily_ret * pos * (pos < -0.1)).sum()
    flat_contrib = (daily_ret * pos * (pos.abs() <= 0.1)).sum()
    bh_missed = (daily_ret * (1 - pos)).sum()  # return we missed vs B&H
    
    print(f"\n  RETURN ATTRIBUTION:")
    print(f"    From long positions:   {long_contrib:.3f}")
    print(f"    From short positions:  {short_contrib:.3f}")
    print(f"    From flat/near-zero:   {flat_contrib:.3f}")
    print(f"    Return missed vs B&H:  {bh_missed:.3f}")
    
    # By regime
    print(f"\n  BY REGIME (avg position, time%):")
    for regime in ['Growth', 'Stagnation', 'Transition', 'Crisis']:
        mask = reg == regime
        if mask.any():
            pct = mask.mean()
            avg_p = pos[mask].mean()
            missed = (daily_ret[mask] * (1 - pos[mask])).sum()
            print(f"    {regime:12s}: pos={avg_p:+.3f}, time={pct:.0%}, return missed={missed:+.3f}")
    
    # If we had just held the V3 average position for ALL bars
    simple_avg = pos.mean()
    hypothetical = (daily_ret * simple_avg).sum()
    print(f"\n  INSIGHT: If held constant pos={simple_avg:+.3f}, return would be ≈{hypothetical:.1%}")
    print(f"  → Strategy variability {'helps' if result.total_return > hypothetical else 'hurts'}")
