"""
V3 MoA Trading Framework — Benchmark Runner
Includes V1/V2/V3 comparison at end.

Usage:
    conda run -n ml python run_benchmark.py
"""

import sys
import os

# Add project root to path for shared modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')
# V3 must be first, then project root, then V1
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, V3_DIR)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader, get_benchmark_tickers
from indicators import add_all_indicators
from backtest_engine import BacktestEngine, BacktestResult

from regime_detector import RegimeDetector
from moa_gating import MoAGatingNetwork, GatingOutput
from moa_ensemble import MoASoftEnsemble, EnsembleOutput
from meta_controller import MetaController, ControllerOutput

# V1 agents from project root
from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.volatility_agent import VolatilityAgent
from agents.crisis_agent import CrisisAgent

# V3 new agent
from v3_agents.exponential_momentum_agent import ExponentialMomentumAgent


class MoAStrategyV3:
    """
    V3 MoA Trading Strategy.
    
    Key differences from V2:
    - Regime-aware position bias (Growth floor, Stagnation tilt)
    - Asymmetric sizing (1.3× longs, 0.7× shorts)
    - Momentum memory in ensemble
    - Trend confirmation bonus in gating
    - Boosted ExponentialMomentumAgent signals
    """
    
    def __init__(
        self,
        top_k: int = 3,
        gating_temperature: float = 0.8,
        conflict_threshold: float = 0.8,
        transaction_cost: float = 0.001,
        debug: bool = True
    ):
        self.regime_detector = RegimeDetector()
        
        self.agents = {
            'TrendAgent': TrendAgent(),
            'MeanReversionAgent': MeanReversionAgent(),
            'VolatilityAgent': VolatilityAgent(),
            'CrisisAgent': CrisisAgent(),
            'ExponentialMomentumAgent': ExponentialMomentumAgent(),
        }
        
        self.gating = MoAGatingNetwork(
            top_k=top_k,
            temperature=gating_temperature,
        )
        
        self.ensemble = MoASoftEnsemble(
            conflict_threshold=conflict_threshold,
        )
        
        self.controller = MetaController(
            transaction_cost=transaction_cost,
        )
        
        self.debug = debug
        self.debug_log: List[Dict] = []
        self._agent_contribution_sum: Dict[str, float] = {name: 0.0 for name in self.agents}
        self._agent_activation_count: Dict[str, int] = {name: 0 for name in self.agents}
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, str]:
        """Generate trading signal for the current bar."""
        
        current_data = df.iloc[:current_idx + 1]
        
        if len(current_data) < 60:
            return 0.0, 'Unknown'
        
        # Get regime probabilities
        regime_name, regime_probs = self.regime_detector.get_current_regime(current_data)
        
        if not regime_probs:
            return 0.0, regime_name
        
        # Current volatility
        if 'rolling_volatility' in df.columns:
            current_vol = df['rolling_volatility'].iloc[current_idx]
            if pd.isna(current_vol) or current_vol <= 0:
                current_vol = 0.20
        else:
            current_vol = 0.20
        
        # Compute gating weights
        gating_output = self.gating.compute_weights(regime_probs)
        
        # Get signals from all agents
        agent_signals = {}
        for agent_name in gating_output.active_agents:
            if agent_name in self.agents:
                signal = self.agents[agent_name].generate_signal(current_data, regime_probs)
                agent_signals[agent_name] = signal
                
                # Track contributions
                self._agent_activation_count[agent_name] += 1
                self._agent_contribution_sum[agent_name] += abs(signal.action * signal.confidence)
        
        if not agent_signals:
            return 0.0, regime_name
        
        # Combine via ensemble
        ensemble_output = self.ensemble.combine_signals(
            agent_signals,
            gating_output.weights,
            current_volatility=current_vol
        )
        
        # Compute position via meta-controller
        controller_output = self.controller.compute_position(
            ensemble_output.final_action,
            ensemble_output.confidence,
            current_vol,
            regime_probs
        )
        
        # Execute trade
        if controller_output.is_trade_allowed:
            prev_pos = self.controller.current_position
            self.controller.execute_trade(controller_output.target_position)
            
            # Update equity
            if 'returns' in df.columns and current_idx > 0:
                daily_ret = df['returns'].iloc[current_idx]
                if not pd.isna(daily_ret):
                    portfolio_ret = prev_pos * daily_ret
                    self.controller.update_equity(portfolio_ret)
        
        # Debug logging (every 50 bars)
        if self.debug and current_idx % 50 == 0:
            date_str = str(df.index[current_idx])[:10]
            active_str = ', '.join(gating_output.active_agents)
            
            sig_details = []
            for name, sig in agent_signals.items():
                w = gating_output.weights.get(name, 0)
                sig_details.append(f"    {name}: action={sig.action:.3f}, conf={sig.confidence:.2f}, w={w:.2f}")
            
            biased_action = controller_output.metadata.get('biased_action', ensemble_output.final_action)
            
            self.debug_log.append({
                'date': date_str,
                'regime': regime_name,
                'position': controller_output.target_position,
                'agents': active_str,
                'ensemble_action': ensemble_output.final_action,
                'biased_action': biased_action,
                'confidence': ensemble_output.confidence,
                'risk_budget': controller_output.risk_budget,
            })
            
            print(f"    [{date_str}] Regime: {regime_name}, Pos: {controller_output.target_position:.2f}, "
                  f"EnsAct: {ensemble_output.final_action:.3f}, BiasAct: {biased_action:.3f}, "
                  f"Conf: {ensemble_output.confidence:.2f}")
            for detail in sig_details:
                print(detail)
        
        return controller_output.target_position, regime_name
    
    def get_agent_summary(self) -> str:
        """Print per-agent contribution summary."""
        lines = ["  Per-Agent Summary:"]
        for name in self.agents:
            count = self._agent_activation_count[name]
            avg_contrib = self._agent_contribution_sum[name] / max(1, count)
            lines.append(f"    {name}: active {count} bars, avg |action*conf|={avg_contrib:.3f}")
        return '\n'.join(lines)
    
    def reset(self):
        """Reset for new backtest."""
        self.controller.reset()
        self.ensemble._action_history.clear()
        self.debug_log.clear()
        self._agent_contribution_sum = {name: 0.0 for name in self.agents}
        self._agent_activation_count = {name: 0 for name in self.agents}


def plot_results(
    ticker: str, 
    result: BacktestResult,
    save_dir: str = '../Plots'
):
    """Generate and save performance plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'V3 MoA Strategy: {ticker}', fontsize=16, fontweight='bold')
    
    # 1. Portfolio vs Benchmark
    ax = axes[0]
    ax.plot(result.equity_curve.index, result.equity_curve.values, 
            label=f'MoA V3 ({result.total_return:.1%})', linewidth=2, color='#2196F3')
    ax.plot(result.benchmark_curve.index, result.benchmark_curve.values,
            label=f'Buy & Hold ({result.benchmark_return:.1%})', linewidth=1.5, 
            color='#FF9800', linestyle='--')
    ax.set_ylabel('Portfolio Value')
    ax.legend(loc='upper left')
    ax.set_title('Equity Curve')
    ax.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax = axes[1]
    ax.fill_between(result.drawdown_curve.index, 0, result.drawdown_curve.values,
                     color='#F44336', alpha=0.4)
    ax.set_ylabel('Drawdown')
    ax.set_title(f'Drawdown (Max: {result.max_drawdown:.1%})')
    ax.grid(True, alpha=0.3)
    
    # 3. Position History
    ax = axes[2]
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                     where=result.position_history >= 0, color='#4CAF50', alpha=0.5, label='Long')
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                     where=result.position_history < 0, color='#F44336', alpha=0.5, label='Short')
    ax.set_ylabel('Position (%)')
    ax.set_title('Position History')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # 4. Regime
    ax = axes[3]
    regime_colors = {
        'Growth': '#4CAF50',
        'Stagnation': '#9E9E9E',
        'Transition': '#FF9800',
        'Crisis': '#F44336',
        'Unknown': '#BDBDBD'
    }
    
    for regime_name, color in regime_colors.items():
        mask = result.regime_history == regime_name
        if mask.any():
            ax.fill_between(result.regime_history.index, 0, 1,
                           where=mask, color=color, alpha=0.4, label=regime_name)
    
    ax.set_ylabel('Regime')
    ax.set_title('Regime History')
    ax.legend(loc='upper right', ncol=4)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'v3_{ticker}_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(all_results: Dict[str, BacktestResult], save_dir: str = '../Plots'):
    """Generate summary comparison plot."""
    os.makedirs(save_dir, exist_ok=True)
    
    tickers = list(all_results.keys())
    strategy_returns = [all_results[t].total_return * 100 for t in tickers]
    benchmark_returns = [all_results[t].benchmark_return * 100 for t in tickers]
    alphas = [all_results[t].alpha * 100 for t in tickers]
    sharpes = [all_results[t].sharpe_ratio for t in tickers]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('V3 MoA Framework — Benchmark Summary', fontsize=16, fontweight='bold')
    
    # Returns comparison
    ax = axes[0]
    x = np.arange(len(tickers))
    width = 0.35
    ax.bar(x - width/2, strategy_returns, width, label='MoA V3', color='#2196F3')
    ax.bar(x + width/2, benchmark_returns, width, label='Buy & Hold', color='#FF9800')
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Returns')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Alpha
    ax = axes[1]
    colors = ['#4CAF50' if a >= 0 else '#F44336' for a in alphas]
    ax.bar(tickers, alphas, color=colors)
    ax.set_ylabel('Alpha (%)')
    ax.set_title('Alpha')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Sharpe
    ax = axes[2]
    colors = ['#4CAF50' if s >= 0 else '#F44336' for s in sharpes]
    ax.bar(tickers, sharpes, color=colors)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'v3_benchmark_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def run_single_backtest(
    ticker: str,
    period: str = '2y',
    debug: bool = True
) -> Optional[BacktestResult]:
    """Run backtest for a single ticker."""
    
    print(f"\n{'='*60}")
    print(f"Running V3 backtest for {ticker}")
    print(f"{'='*60}")
    
    loader = DataLoader()
    
    print("  Fetching data...")
    try:
        df = loader.fetch_ticker(ticker, period=period)
    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")
        return None
    
    print("  Computing indicators...")
    df = add_all_indicators(df)
    
    strategy = MoAStrategyV3(debug=debug)
    
    print("  Training regime detector...")
    strategy.regime_detector.fit(df)
    
    print("  Running backtest...")
    engine = BacktestEngine(transaction_cost=0.001)
    
    try:
        result = engine.run(df, strategy.generate_signal, warmup_period=60)
    except Exception as e:
        print(f"  ERROR during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"\n  Results for {ticker}:")
    print(f"    Total Return:   {result.total_return:>8.2%}")
    print(f"    Benchmark:      {result.benchmark_return:>8.2%}")
    print(f"    Alpha:          {result.alpha:>8.2%}")
    print(f"    Sharpe:         {result.sharpe_ratio:>8.2f}")
    print(f"    Max Drawdown:   {result.max_drawdown:>8.2%}")
    print(f"    Num Trades:     {result.num_trades:>8d}")
    
    print(strategy.get_agent_summary())
    
    # Position analysis
    pos = result.position_history
    reg = result.regime_history
    combined = pd.DataFrame({'position': pos, 'regime': reg})
    growth_pos = combined[combined['regime']=='Growth']['position'].mean() if (reg == 'Growth').any() else 0
    near_zero_pct = (pos.abs() < 0.1).mean()
    print(f"  Position Analysis:")
    print(f"    Growth avg pos:     {growth_pos:>+.3f}")
    print(f"    Mean |pos|:         {pos.abs().mean():>.3f}")
    print(f"    Time near-zero:     {near_zero_pct:>.1%}")
    
    plot_results(ticker, result)
    print(f"  Plots saved to Plots/")
    
    return result


def run_full_benchmark(
    tickers: Optional[List[str]] = None,
    period: str = '2y',
    debug: bool = True
):
    """Run full benchmark across all tickers."""
    
    if tickers is None:
        tickers = get_benchmark_tickers()
    
    print(f"\n{'='*60}")
    print(f"MoA V3 Trading Framework — Full Benchmark")
    print(f"{'='*60}")
    print(f"Tickers: {tickers}")
    print(f"Period: {period}")
    print(f"{'='*60}")
    
    all_results = {}
    
    for ticker in tickers:
        result = run_single_backtest(ticker, period, debug)
        if result is not None:
            all_results[ticker] = result
    
    if all_results:
        print(f"\n{'='*60}")
        print(f"V3 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Ticker':>6}  {'Total Return':>12}  {'Benchmark':>12}  {'Alpha':>8}  {'Sharpe':>8}  "
              f"{'Sortino':>8}  {'Max DD':>8}  {'Calmar':>8}  {'Trades':>8}  {'Win Rate':>8}")
        
        for ticker, r in all_results.items():
            print(f"{ticker:>6}  {r.total_return:>12.4%}  {r.benchmark_return:>12.4%}  "
                  f"{r.alpha:>8.4%}  {r.sharpe_ratio:>8.2f}  {r.sortino_ratio:>8.2f}  "
                  f"{r.max_drawdown:>8.4%}  {r.calmar_ratio:>8.2f}  "
                  f"{r.num_trades:>8d}  {r.win_rate:>8.2%}")
        
        avg_alpha = np.mean([r.alpha for r in all_results.values()])
        avg_sharpe = np.mean([r.sharpe_ratio for r in all_results.values()])
        avg_dd = np.mean([r.max_drawdown for r in all_results.values()])
        avg_return = np.mean([r.total_return for r in all_results.values()])
        avg_benchmark = np.mean([r.benchmark_return for r in all_results.values()])
        
        print(f"\n{'-'*60}")
        print(f"AVERAGES:")
        print(f"  Avg Return:          {avg_return:.2%}")
        print(f"  Avg Benchmark:       {avg_benchmark:.2%}")
        print(f"  Avg Alpha:           {avg_alpha:.2%}")
        print(f"  Avg Sharpe:          {avg_sharpe:.2f}")
        print(f"  Avg Max Drawdown:    {avg_dd:.2%}")
        print(f"  Return Capture:      {avg_return/avg_benchmark:.0%} of B&H")
        
        plot_summary(all_results)
        print(f"\n  Summary plot saved to Plots/v3_benchmark_summary.png")
    
    return all_results


if __name__ == '__main__':
    results = run_full_benchmark()
