# Meta Hierarchic MARS Strategy

## Introduction

The **Meta Hierarchic MARS (Meta-Adaptive Reinforcement Learning System) Strategy** is a sophisticated quantitative trading framework designed to solve the problem of market non-stationarity. Traditional single-logic strategies (e.g., pure trend following or mean reversion) inevitably fail when market regimes shift. This framework employs a **Hierarchical Mixture-of-Experts (H-MoE)** architecture, where specialized agents (Trend, Mean Reversion, Volatility, Hedge) are dynamically allocated capital by a "Meta-Controller" based on the prevailing market regime.

The strategy recognizes four distinct market phases:

1.  **Growth**: High drift, low volatility (Trend Following).
2.  **Stagnation**: Mean reverting, low volatility (Liquidity Provision).
3.  **Crisis**: High volatility, negative skew (Tail-Risk Hedging/Shorting).
4.  **Transition**: Rising volatility, directionless (Breakout/Long Volatility).

By adapting to these regimes in real-time, the MARS strategy aims to capture "Crisis Alpha" while minimizing drawdowns during choppy transitions.

---

## Strategy Evolution & Analysis (v3 - v5)

The codebase tracks the evolution of this system from a conservative prototype to a high-octane alpha generator, and finally to a robust, institutional-grade risk-adjusted engine.

### v3: Conservative Alpha

**"The Stable Baseline"**
Version 3 established the core stability of the hierarchical model. It prioritized capital preservation and high Sharpe ratios over raw explosive returns.

*   **Characteristics**: Extremely low Maximum Drawdown (Avg ~5.7%) and high risk-adjusted returns (Sharpe ~1.74).
*   **Behavior**: Quick to de-lever in uncertainty. While it underperformed in raging bull markets compared to pure long-only strategies, it offered superior protection during corrections.

### v4: Max Return

**"The Convexity Engine"**
Version 4 was engineered to maximize total capture. It loosened the risk constraints to allow the agents to ride trends more aggressively.

*   **Characteristics**: Massive raw returns (Avg ~88.1%) and high Buy & Hold capture (112%).
*   **Trade-off**: The pursuit of alpha came at the cost of volatility. Average Max Drawdown spiked to ~18%, with volatile assets like TSLA seeing up to 36% drawdowns. It effectively beat the market but with higher variance.

### v5: Risk-Adjusted Precision

**"The Smart Defense"**
Version 5 represents the maturation of the strategy. It introduces a multi-layered defense system to curb the drawdowns of v4 without sacrificing its alpha-generating capability. It implements three specific mechanisms:

1.  **Regime Momentum**: Instead of reacting to instant signals, v5 tracks the momentum of regime probabilities over a 5-bar window, providing a 3-5 day early warning system for regime transitions.
2.  **VIX Fear Gauge**: Incorporates external market fear data (via `^VIX`). When the VIX > 25, the baseline exposure is automatically engaged, overriding local signals to protect capital.
3.  **Drawdown-Adaptive Baseline**: A dynamic feedback loop that scales down position sizing as drawdowns deepen, explicitly capping maximum loss potential.

---

## Performance Comparison

The following benchmarks illustrate the progression of the strategy across five major technology tickers.

### Ticker Performance (Total Return)

| Ticker   | v1   | v2   | v3       | v4        | v5        | Buy & Hold |
| :------- | :--- | :--- | :------- | :-------- | :-------- | :--------- |
| **AAPL** | +14% | +31% | +41%     | +48%      | +51%      | **+52%**   |
| **MSFT** | -0%  | +14% | **+25%** | +12%      | +18%      | +2%        |
| **GOOG** | +28% | +26% | +41%     | +74%      | +69%      | **+91%**   |
| **NVDA** | -16% | +42% | +53%     | +125%     | **+138%** | +110%      |
| **TSLA** | +36% | +62% | +135%    | **+181%** | +103%     | +139%      |

> **Note**: v5 achieves superior returns on NVDA (+138%) compared to both v4 and Buy & Hold, while significantly smoothing out the equity curve.

### Aggregate Metrics (v3 - v5)

| Metric          | v3        | v4        | v5    |
| :-------------- | :-------- | :-------- | :---- |
| **Avg Return**  | 58.9%     | **88.1%** | 75.7% |
| **B&H Capture** | 75%       | **112%**  | 96%   |
| **Avg Alpha**   | **24.4%** | 16.1%     | 16.5% |
| **Avg Sharpe**  | **1.74**  | 1.34      | 1.42  |
| **Avg MaxDD**   | **5.7%**  | 18.0%     | 16.0% |

### Key Takeaway: The v4 vs v5 Trade-off
*   **Choose v4** if you want **Maximum Raw Return** and can stomach ~18-36% drawdowns. It is the "risk-on" version of the strategy.
*   **Choose v5** if you want **Risk-Adjusted Consistency**. It trades a small portion of the raw upside (Avg Return 88% -> 76%) to significantly improve the safety profile and reliability of the strategy.
