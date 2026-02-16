import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIO_V2_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for v1 dir for imports
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')

for p in [CIO_V2_DIR, V1_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cio_v2.run_benchmark import run_cio_benchmark, CIO_TICKERS
from cio_v2.cio_orchestrator import CIOResult

st.set_page_config(page_title="CIO Agent Dashboard", layout="wide")

st.title("🤖 CIO Agent Performance Dashboard")
st.markdown("""
This dashboard visualizes the performance of the **CIO Agent (v2 Logic + V3 Mean-Variance Allocator)**.
It demonstrates how the agent allocates capital across a portfolio of tech stocks based on:
1.  **Base Model Signals**: Trend, Regime, and Volatility signals from the v7-improved model.
2.  **Portfolio Optimization**: Mean-Variance optimization to maximize Sharpe ratio.
3.  **Risk Management**: Dynamic drawdown control and volatility targeting.
""")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")
allocator_choice = st.sidebar.selectbox("Allocator", ["v3", "v2", "v1"], index=0, format_func=lambda x: f"{x.upper()} Allocator")
period = st.sidebar.selectbox("Backtest Period", ["1y", "2y", "5y"], index=1)
run_btn = st.sidebar.button("Run Benchmark")

# Cached function to run benchmark
@st.cache_data(show_spinner=True)
def get_benchmark_results(allocator, period):
    # Capture stdout to avoid cluttering UI
    # We invoke the run_cio_benchmark function directly
    # Note: run_benchmark prints to stdout, we might want to suppress it or capture it
    results = run_cio_benchmark(allocator_choice=allocator, period=period)
    return results

if run_btn:
    with st.spinner(f"Running CIO benchmark with {allocator_choice.upper()} allocator..."):
        try:
            results_map = get_benchmark_results(allocator_choice, period)
            
            # Extract the correct result object based on the key used in run_benchmark
            # The keys in run_benchmark are 'V3 Mean-Variance', etc.
            # We need to map 'v3' -> 'V3 Mean-Variance'
            key_map = {
                'v1': 'V1 Risk Parity',
                'v2': 'V2 Signal-Weighted',
                'v3': 'V3 Mean-Variance'
            }
            target_key = key_map.get(allocator_choice)
            
            if target_key in results_map:
                st.session_state['result'] = results_map[target_key]
                st.session_state['allocator_name'] = target_key
            else:
                st.error(f"Results for {target_key} not found. keys: {list(results_map.keys())}")
                
        except Exception as e:
            st.error(f"Error running benchmark: {e}")
            st.exception(e)

if 'result' in st.session_state:
    result: CIOResult = st.session_state['result']
    allocator_name = st.session_state['allocator_name']
    
    # -------------------------------------------------------------------------
    # Key Metrics
    # -------------------------------------------------------------------------
    st.header(f"Performance Summary: {allocator_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Return", f"{result.total_return:.2%}", delta=f"{result.total_return - result.equal_weight_return:.2%} vs B&H")
    col2.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    col3.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
    col4.metric("Avg Cash", f"{result.avg_cash_pct:.1%}")
    col5.metric("Sortino", f"{result.sortino_ratio:.2f}")

    # -------------------------------------------------------------------------
    # Tabs for detailed view
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Equity & Drawdown", "Allocation & Composition", "Ticker Deep Dive", "Allocator Internals"])
    
    with tab1:
        st.subheader("Equity Curve")
        
        # Plotly Equity Curve
        df_equity = pd.DataFrame({
            "CIO Strategy": result.equity_curve,
            "Equally Weighted B&H": (1 + result.equal_weight_return) # This is a scalar in the result object, we ideally need the curve
            # The result object has equity_curve (Series). It doesn't seem to store the B&H curve explicitly, just the final scalar?
            # Looking at plot_cio_results, it plots a Horizontal line for B&H return? 
            # Ah, "ax.axhline(y=1 + result.equal_weight_return..." -> It plots the FINAL return as a target line.
            # We'll replicate that.
        })
        
        fig_eq = px.line(result.equity_curve, title="Portfolio Value Over Time")
        fig_eq.add_trace(go.Scatter(
            x=[result.equity_curve.index[0], result.equity_curve.index[-1]],
            y=[1+result.equal_weight_return, 1+result.equal_weight_return],
            mode="lines", name="Buy & Hold (Final)", line=dict(dash="dash", color="orange")
        ))
        fig_eq.update_layout(yaxis_title="Normalized Value", xaxis_title="Date")
        st.plotly_chart(fig_eq, use_container_width=True)
        
        st.subheader("Drawdown")
        fig_dd = px.area(result.drawdown_curve * -100, title="Drawdown (%)") # Drawdown is usually positive in calculation (peak-val)/peak, make it negative for plot?
        # The stored drawdown_curve seems to be positive values (e.g. 0.10 for 10% DD). 
        # Visualization usually shows underwater plot (negative).
        fig_dd.update_traces(fillcolor="red", line_color="red")
        fig_dd.update_yaxes(autorange="reversed") # Or just plot as negative
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab2:
        st.subheader("Portfolio Allocation Over Time")
        
        # Stacked Area Chart
        wt_df = result.weight_history
        if not wt_df.empty:
            # Add Cash to weights for completeness
            # wt_df does not contain cash column by default, let's create a display df
            df_area = wt_df.copy()
            df_area['Cash'] = result.cash_history
            
            fig_area = px.area(df_area, title="Asset Allocation History", groupnorm=None)
            st.plotly_chart(fig_area, use_container_width=True)
        
        st.subheader("Allocation Heatmap")
        # Transpose for Heatmap: Tickers on Y, Time on X
        if not wt_df.empty:
            # Resample to weekly to reduce data points for heatmap
            wt_weekly = wt_df.resample('W').mean().T
            fig_heat = px.imshow(wt_weekly, aspect="auto", color_continuous_scale="Greens", title="Allocation Intensity (Weekly Avg)")
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.subheader("Single Ticker Analysis")
        ticker = st.selectbox("Select Ticker", CIO_TICKERS)
        
        if ticker in result.ticker_returns:
            # We don't have price data in the result object directly, only weights/signals/returns.
            # Realistically we should access the data_loader or store price in result for viz.
            # But we can visualize the WEIGHT assigned to this ticker.
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric(f"{ticker} Contribution", f"{result.ticker_contributions.get(ticker, 0):.2%}")
            with col_t2:
                st.metric(f"{ticker} B&H Return", f"{result.ticker_returns.get(ticker, 0):.2%}")
            
            # Plot Weight vs Time
            if ticker in result.weight_history.columns:
                st.line_chart(result.weight_history[ticker])
                st.caption(f"Allocation Weight for {ticker}")
                
            # If we had regime history...
            if hasattr(result, 'regime_history') and not result.regime_history.empty:
                 if ticker in result.regime_history.columns:
                     st.write(f"Regime History for {ticker}")
                     st.dataframe(result.regime_history[ticker].value_counts())

    with tab4:
        st.subheader("How It Works: Mean-Variance Optimization")
        st.markdown(r"""
        The **Mean-Variance Allocator (V3)** optimizes weights by solving the classic Markowitz problem:
        
        $$
        \max \mathbf{w}^T \boldsymbol{\mu} - \lambda \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}
        $$
        
        Where:
        - $\mathbf{w}$ = Vector of weights (what we want to find)
        - $\boldsymbol{\mu}$ = Expected returns vector (derived from **Base Model Signals**)
        - $\mathbf{\Sigma}$ = Covariance matrix (derived from **Price History**)
        - $\lambda$ = Risk aversion parameter
        
        **Why it beats v2:**
        It accounts for **correlations**. If two stocks (e.g., NVDA and AMD) are highly correlated and volatile, 
        V3 will penalize holding both simultaneously more than V2 would, leading to better diversification.
        """)
        
        st.info("To see the actual Covariance Matrix and Expected Returns for a specific date, we would need to capture the allocator's internal state during the backtest. Currently, this dashboard shows the post-hoc results.")
        
        # We could potentially re-compute it here if we loaded the data, but for now we explain the logic.
        
else:
    st.info("Please click 'Run Benchmark' in the sidebar to start.")
