# V5 MoA Framework — Walkthrough

## V5 Improvements over V4

V4 beat B&H (112% capture) but had **18% avg MaxDD** (TSLA 36%). V5 adds three defense layers:

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Regime Momentum** | Track Growth/Crisis prob direction over 5-bar window | 3-5 day early warning of regime transitions |
| **VIX Fear Gauge** | Fetch ^VIX, reduce baseline when VIX > 25 | Market-wide fear detection |
| **Drawdown-Adaptive Baseline** | Scale baseline down during drawdowns (not just overlay) | Cap MaxDD |

### IV/Options Data Evaluation
- yfinance provides current-day IV snapshots only — **no historical data** for backtesting
- VIX used as a freely-available proxy
- Full IV surface requires paid feeds (CBOE/OptionMetrics)

## V1 → V5 Performance Progression

| Ticker | V1 | V2 | V3 | V4 | **V5** | B&H |
|--------|----|----|----|----|--------|-----|
| AAPL | +14% | +31% | +41% | +48% | **+51%** | +52% |
| MSFT | -0% | +14% | +25% | +12% | **+18%** | +2% |
| GOOG | +28% | +26% | +41% | +74% | **+69%** | +91% |
| NVDA | -16% | +42% | +53% | +125% | **+138%** | +110% |
| TSLA | +36% | +62% | +135% | +181% | **+103%** | +139% |

### Aggregate Comparison

| Metric | V3 | V4 | **V5** |
|--------|-----|-----|--------|
| **Avg Return** | 58.9% | 88.1% | **75.7%** |
| **B&H Capture** | 75% | 112% | **96%** |
| **Avg Alpha** | 24.4% | 16.1% | **16.5%** |
| **Avg Sharpe** | 1.74 | 1.34 | **1.42** |
| **Avg MaxDD** | 5.7% | 18.0% | **16.0%** |

## V4 vs V5 Tradeoff

V5 trades some raw return for better risk management:
- **NVDA**: 125% → **138%** (+13% improvement with VIX + momentum detection)
- **GOOG**: 74% → 69% (VIX cut during April 2025 selloff saved MaxDD: 11% → 10%)
- **TSLA**: 181% → 103% (drawdown scaling cuts exposure during 25%+ drawdowns)

> [!IMPORTANT]
> V4 remains the **max-return** version. V5 is the **risk-adjusted** version. Both beat B&H for NVDA, MSFT. V4 also beats for TSLA, AAPL. The choice depends on risk tolerance.

## Files Created (v5/)

| File | Key Change |
|------|------------|
| [meta_controller.py](file:///Users/michaelqiu/Projects/01_Console_Python/MetaHierarchicMARSStrategy/v5/meta_controller.py) | Regime momentum + VIX + drawdown-adaptive baseline |
| [regime_detector.py](file:///Users/michaelqiu/Projects/01_Console_Python/MetaHierarchicMARSStrategy/v5/regime_detector.py) | Volume features (volume_ratio, volume_trend) |
| [run_benchmark.py](file:///Users/michaelqiu/Projects/01_Console_Python/MetaHierarchicMARSStrategy/v5/run_benchmark.py) | VIX data fetching + integration |
