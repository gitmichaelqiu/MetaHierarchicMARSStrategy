"""
Benchmark Runner for MoA Trading Framework
Tests the Mixture of Agents strategy on technology tickers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader, get_benchmark_tickers
from indicators import add_all_indicators
from regime_detector import RegimeDetector, add_regime_features
from agents import TrendAgent, MeanReversionAgent, VolatilityAgent, CrisisAgent, ExponentialMomentumAgent, AgentSignal
from moa_gating import MoAGatingNetwork
from moa_ensemble import MoASoftEnsemble
from meta_controller import MetaController
from backtest_engine import BacktestEngine, BacktestResult


class MoAStrategy:
    """
    Complete Mixture of Agents trading strategy.
    
    Integrates:
    - Regime detection (GMM-based)
    - 4 specialized agents
    - Gating network (Top-K selection)
    - Soft-ensemble voting
    - Meta-controller for risk management
    """
    
    def __init__(
        self,
        top_k: int = 2,
        gating_temperature: float = 1.0,
        conflict_threshold: float = 0.6,  # Higher threshold - less penalty
        transaction_cost: float = 0.001,
        kelly_fraction: float = 1.0,  # Full Kelly for more aggressive sizing
        retrain_freq: int = 60,
        train_window: int = 252,
        debug: bool = True
    ):
        """
        Initialize the MoA strategy.
        
        Args:
            top_k: Number of agents to select
            gating_temperature: Temperature for gating softmax
            conflict_threshold: Threshold for conflict detection
            transaction_cost: Transaction cost per trade
            kelly_fraction: Fraction of Kelly criterion for sizing
            debug: Print debug information
        """
        self.debug = debug
        
        # Initialize components
        self.regime_detector = RegimeDetector()
        
        self.agents = {
            'TrendAgent': TrendAgent(),
            'MeanReversionAgent': MeanReversionAgent(),
            'VolatilityAgent': VolatilityAgent(),
            'CrisisAgent': CrisisAgent(),
            'ExponentialMomentumAgent': ExponentialMomentumAgent()
        }
        
        self.gating = MoAGatingNetwork(
            top_k=top_k,
            temperature=gating_temperature
        )
        
        self.ensemble = MoASoftEnsemble(
            conflict_threshold=conflict_threshold
        )
        
        # Use defaults from MetaController for more aggressive positioning
        self.controller = MetaController(
            transaction_cost=transaction_cost,
            max_position=2.0  # Allow leverage
        )
        
        self.retrain_freq = retrain_freq
        self.train_window = train_window
        self.last_train_idx = 0
        
        # State
        self.is_fitted = False
        self.debug_history: List[Dict] = []
    
    def fit(self, df: pd.DataFrame) -> 'MoAStrategy':
        """Fit the regime detector on training data."""
        print("  Training regime detector...")
        self.regime_detector.fit(df)
        self.is_fitted = True
        self.last_train_idx = len(df)
        return self
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, str]:
        """
        Generate trading signal for the current bar.
        
        Args:
            df: Full DataFrame with OHLCV and indicators
            current_idx: Current bar index
            
        Returns:
            (position_pct, regime_name)
        """
        if not self.is_fitted:
            return 0.0, 'Unknown'
        
        # Slice data up to current bar (no lookahead)
        current_data = df.iloc[:current_idx + 1]
        
        if len(current_data) < 50:
            return 0.0, 'Warmup'
            
        # Check for retraining
        if self.is_fitted and (current_idx - self.last_train_idx >= self.retrain_freq):
            # Retrain on rolling window
            train_start = max(0, current_idx - self.train_window)
            train_data = df.iloc[train_start:current_idx]
            
            # Ensure we have enough data and volatility isn't zero
            if len(train_data) > 100:
                try:
                    self.regime_detector.fit(train_data)
                    self.last_train_idx = current_idx
                    if self.debug:
                        print(f"    [Retrained Regime Detector at idx {current_idx}]")
                except Exception as e:
                    if self.debug:
                        print(f"    [Retrain Failed: {e}]")
        
        # Get regime probabilities
        try:
            regime, regime_probs = self.regime_detector.get_current_regime(current_data)
        except Exception as e:
            return 0.0, 'Error'
        
        # Get current volatility
        if 'hist_vol' in current_data.columns:
            current_vol = current_data['hist_vol'].iloc[-1]
            if pd.isna(current_vol):
                current_vol = 0.25
        else:
            current_vol = 0.25
        
        # Compute gating weights
        gating_output = self.gating.compute_weights(regime_probs)
        
        # Get signals from all agents
        agent_signals = {}
        for name, agent in self.agents.items():
            try:
                signal = agent.generate_signal(current_data, regime_probs)
                agent_signals[name] = signal
            except Exception as e:
                agent_signals[name] = AgentSignal(action=0, confidence=0, regime_fit=0)
        
        # Combine signals with ensemble
        ensemble_output = self.ensemble.combine_signals(
            agent_signals,
            gating_output.weights,
            current_volatility=current_vol
        )
        
        # Apply meta-controller
        controller_output = self.controller.compute_position(
            ensemble_output.final_action,
            ensemble_output.confidence,
            current_vol,
            regime_probs
        )
        
        # Execute trade in controller
        if controller_output.is_trade_allowed:
            self.controller.execute_trade(controller_output.target_position)
        
        # Debug output
        if self.debug:
            debug_info = {
                'date': current_data.index[-1],
                'regime': regime,
                'regime_probs': regime_probs,
                'active_agents': gating_output.active_agents,
                'agent_weights': gating_output.weights,
                'ensemble_action': ensemble_output.final_action,
                'is_conflicting': ensemble_output.is_conflicting,
                'conflict_penalty': ensemble_output.conflict_penalty,
                'target_position': controller_output.target_position,
                'risk_budget': controller_output.risk_budget
            }
            self.debug_history.append(debug_info)
            
            # Print occasionally
            if len(self.debug_history) % 50 == 0:
                print(f"    [{current_data.index[-1].strftime('%Y-%m-%d')}] "
                      f"Regime: {regime}, Pos: {controller_output.target_position:.2f}, "
                      f"Agents: {gating_output.active_agents}")
        
        return controller_output.target_position, regime
    
    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        self.controller.reset()
        for agent in self.agents.values():
            agent.signal_history.clear()
        self.debug_history.clear()


def run_single_ticker_backtest(
    ticker: str,
    period: str = '2y',
    debug: bool = True
) -> Tuple[BacktestResult, MoAStrategy]:
    """
    Run backtest on a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Historical period to fetch
        debug: Print debug information
        
    Returns:
        (BacktestResult, strategy)
    """
    print(f"\n{'='*60}")
    print(f"Running backtest for {ticker}")
    print('='*60)
    
    # Load data
    print(f"  Fetching data...")
    loader = DataLoader()
    df = loader.fetch_ticker(ticker, period=period)
    
    # Add indicators
    print(f"  Computing indicators...")
    df = add_all_indicators(df)
    
    # Initialize strategy
    strategy = MoAStrategy(debug=debug)
    
    # Use first portion for training regime detector
    train_split = int(len(df) * 0.3)
    train_data = df.iloc[:train_split]
    strategy.fit(train_data)
    
    # Create signal function
    def signal_func(df, idx):
        return strategy.generate_signal(df, idx)
    
    # Run backtest
    print(f"  Running backtest...")
    engine = BacktestEngine()
    result = engine.run(df, signal_func, warmup_period=train_split)
    
    print(f"\n  Results for {ticker}:")
    print(f"    Total Return:  {result.total_return:>8.2%}")
    print(f"    Benchmark:     {result.benchmark_return:>8.2%}")
    print(f"    Alpha:         {result.alpha:>8.2%}")
    print(f"    Sharpe:        {result.sharpe_ratio:>8.2f}")
    print(f"    Max Drawdown:  {result.max_drawdown:>8.2%}")
    print(f"    Num Trades:    {result.num_trades:>8d}")
    
    return result, strategy


def plot_results(
    ticker: str,
    result: BacktestResult,
    save_dir: str = 'Plots/v2'
) -> None:
    """
    Generate and save visualization plots.
    
    Args:
        ticker: Ticker symbol for titles
        result: Backtest result
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Portfolio vs Buy & Hold
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(result.equity_curve.index, result.equity_curve.values, 
            label='MoA Strategy', linewidth=2, color='#2196F3')
    ax.plot(result.benchmark_curve.index, result.benchmark_curve.values,
            label='Buy & Hold', linewidth=2, color='#9E9E9E', linestyle='--')
    
    ax.fill_between(result.equity_curve.index, 
                    result.equity_curve.values, 
                    result.benchmark_curve.values,
                    where=result.equity_curve.values >= result.benchmark_curve.values,
                    alpha=0.3, color='green', label='Outperformance')
    ax.fill_between(result.equity_curve.index,
                    result.equity_curve.values,
                    result.benchmark_curve.values,
                    where=result.equity_curve.values < result.benchmark_curve.values,
                    alpha=0.3, color='red', label='Underperformance')
    
    ax.set_title(f'{ticker}: MoA Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/{ticker}_portfolio_vs_buyhold.png', dpi=150)
    plt.close()
    
    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.fill_between(result.drawdown_curve.index, 0, result.drawdown_curve.values * 100,
                    color='red', alpha=0.5)
    ax.axhline(y=-result.max_drawdown * 100, color='darkred', linestyle='--',
               label=f'Max DD: {result.max_drawdown:.1%}')
    
    ax.set_title(f'{ticker}: Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/{ticker}_drawdown.png', dpi=150)
    plt.close()
    
    # 3. Position History
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                    where=result.position_history.values >= 0,
                    color='green', alpha=0.5, label='Long')
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                    where=result.position_history.values < 0,
                    color='red', alpha=0.5, label='Short')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax.set_title(f'{ticker}: Position History', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Position (% of Capital)')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/{ticker}_positions.png', dpi=150)
    plt.close()
    
    # 4. Regime History
    if len(result.regime_history) > 0:
        fig, ax = plt.subplots(figsize=(12, 3))
        
        regime_colors = {
            'Growth': 'green',
            'Stagnation': 'gray',
            'Transition': 'orange',
            'Crisis': 'red',
            'Unknown': 'blue',
            'Warmup': 'lightgray',
            'Error': 'black'
        }
        
        regime_nums = {'Growth': 3, 'Stagnation': 2, 'Transition': 1, 'Crisis': 0,
                       'Unknown': -1, 'Warmup': -1, 'Error': -1}
        
        colors = [regime_colors.get(r, 'blue') for r in result.regime_history.values]
        nums = [regime_nums.get(r, -1) for r in result.regime_history.values]
        
        ax.scatter(result.regime_history.index, nums, c=colors, s=5, alpha=0.6)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Crisis', 'Transition', 'Stagnation', 'Growth'])
        
        ax.set_title(f'{ticker}: Detected Regimes', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{save_dir}/{ticker}_regimes.png', dpi=150)
        plt.close()
    
    print(f"  Plots saved to {save_dir}/")


def run_full_benchmark(
    tickers: Optional[List[str]] = None,
    period: str = '2y',
    save_plots: bool = True
) -> pd.DataFrame:
    """
    Run benchmark on multiple tickers.
    
    Args:
        tickers: List of tickers (default: tech tickers)
        period: Historical period
        save_plots: Whether to save visualization plots
        
    Returns:
        DataFrame with results summary
    """
    if tickers is None:
        tickers = get_benchmark_tickers()
    
    print("\n" + "="*60)
    print("MoA Trading Framework - Full Benchmark")
    print("="*60)
    print(f"Tickers: {tickers}")
    print(f"Period: {period}")
    print("="*60)
    
    results = []
    
    for ticker in tickers:
        try:
            result, strategy = run_single_ticker_backtest(ticker, period, debug=True)
            
            if save_plots:
                plot_results(ticker, result)
            
            results.append({
                'Ticker': ticker,
                'Total Return': result.total_return,
                'Benchmark Return': result.benchmark_return,
                'Alpha': result.alpha,
                'Sharpe': result.sharpe_ratio,
                'Sortino': result.sortino_ratio,
                'Max Drawdown': result.max_drawdown,
                'Calmar': result.calmar_ratio,
                'Num Trades': result.num_trades,
                'Win Rate': result.win_rate
            })
            
        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Calculate averages
    if len(summary_df) > 0:
        print("\n" + "-"*60)
        print("AVERAGES:")
        print(f"  Avg Alpha:        {summary_df['Alpha'].mean():>8.2%}")
        print(f"  Avg Sharpe:       {summary_df['Sharpe'].mean():>8.2f}")
        print(f"  Avg Max Drawdown: {summary_df['Max Drawdown'].mean():>8.2%}")
    
    # Save summary plot
    if save_plots and len(summary_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Returns comparison
        ax = axes[0, 0]
        x = np.arange(len(summary_df))
        width = 0.35
        ax.bar(x - width/2, summary_df['Total Return'] * 100, width, label='Strategy', color='#2196F3')
        ax.bar(x + width/2, summary_df['Benchmark Return'] * 100, width, label='Buy & Hold', color='#9E9E9E')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['Ticker'])
        ax.set_ylabel('Return (%)')
        ax.set_title('Returns: Strategy vs Benchmark')
        ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Alpha
        ax = axes[0, 1]
        colors = ['green' if a > 0 else 'red' for a in summary_df['Alpha']]
        ax.bar(summary_df['Ticker'], summary_df['Alpha'] * 100, color=colors, alpha=0.7)
        ax.set_ylabel('Alpha (%)')
        ax.set_title('Alpha (vs Buy & Hold)')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Sharpe
        ax = axes[1, 0]
        ax.bar(summary_df['Ticker'], summary_df['Sharpe'], color='#673AB7', alpha=0.7)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good (>1)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Max Drawdown
        ax = axes[1, 1]
        ax.bar(summary_df['Ticker'], summary_df['Max Drawdown'] * 100, color='#F44336', alpha=0.7)
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Maximum Drawdown')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('Plots/benchmark_summary.png', dpi=150)
        plt.close()
        
        print("\n  Summary plot saved to Plots/benchmark_summary.png")
    
    return summary_df


if __name__ == "__main__":
    # Run the full benchmark
    summary = run_full_benchmark()
