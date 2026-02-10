"""Quick analysis: V2 position sizing gaps during Growth regimes."""
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')
# V2 dir must be first so v2/run_benchmark.py is found before v1/run_benchmark.py
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, V2_DIR)

import pandas as pd, numpy as np
from data_loader import DataLoader
from indicators import add_all_indicators
from run_benchmark import MoAStrategyV2
from backtest_engine import BacktestEngine

loader = DataLoader()

for ticker in ['AAPL', 'TSLA', 'NVDA', 'GOOG']:
    df = loader.fetch_ticker(ticker, period='2y')
    df = add_all_indicators(df)
    
    strategy = MoAStrategyV2(debug=False)
    strategy.regime_detector.fit(df)
    engine = BacktestEngine()
    result = engine.run(df, strategy.generate_signal, warmup_period=60)
    
    pos = result.position_history
    reg = result.regime_history
    combined = pd.DataFrame({'position': pos, 'regime': reg})
    
    print(f"\n{'='*60}")
    print(f"  {ticker}: Return={result.total_return:.1%} vs B&H={result.benchmark_return:.1%}")
    print(f"{'='*60}")
    print(f"  Regime Distribution:")
    for regime, count in reg.value_counts().items():
        pct = count / len(reg)
        avg_pos = combined[combined['regime']==regime]['position'].mean()
        print(f"    {regime:12s}: {count:3d} bars ({pct:.0%}), avg position = {avg_pos:+.3f}")
    
    print(f"\n  Position Stats:")
    print(f"    Mean |position|:          {pos.abs().mean():.3f}")
    print(f"    Time fully invested ≥0.5: {(pos.abs() >= 0.5).mean():.1%}")
    print(f"    Time near zero <0.1:      {(pos.abs() < 0.1).mean():.1%}")
    print(f"    Growth avg position:      {combined[combined['regime']=='Growth']['position'].mean():+.3f}")
    
    # What position would match B&H? 
    # If B&H = 100% long always, strategy needs avg_pos ≈ B&H_return / B&H_return = 1.0
    print(f"    → To match B&H, need avg ~1.0 during Growth")
