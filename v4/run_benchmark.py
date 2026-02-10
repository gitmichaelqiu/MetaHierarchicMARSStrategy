"""
V4 MoA Trading Framework — Benchmark Runner
Adaptive Baseline + Alpha Overlay architecture.

Usage:
    conda run -n ml python run_benchmark.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(PROJECT_ROOT, 'v1')
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, V4_DIR)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader, get_benchmark_tickers
from indicators import add_all_indicators
from backtest_engine import BacktestEngine, BacktestResult

from regime_detector import RegimeDetector
from moa_gating import MoAGatingNetwork, GatingOutput
from moa_ensemble import MoASoftEnsemble, EnsembleOutput
from meta_controller import MetaController, ControllerOutput

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.volatility_agent import VolatilityAgent
from agents.crisis_agent import CrisisAgent
from v4_agents.exponential_momentum_agent import ExponentialMomentumAgent


class MoAStrategyV4:
    """
    V4 MoA: Adaptive Baseline + Alpha Overlay.
    
    Position = regime-weighted baseline + MoA alpha overlay.
    Baseline is always long (0.1-0.8). Overlay adjusts ±0.4.
    """
    
    def __init__(self, debug: bool = True):
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
        
        self.debug = debug
        self.debug_log: List[Dict] = []
        self._agent_activation_count: Dict[str, int] = {n: 0 for n in self.agents}
        self._agent_contribution_sum: Dict[str, float] = {n: 0.0 for n in self.agents}
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, str]:
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
        
        gating_output = self.gating.compute_weights(regime_probs)
        
        agent_signals = {}
        for agent_name in gating_output.active_agents:
            if agent_name in self.agents:
                signal = self.agents[agent_name].generate_signal(current_data, regime_probs)
                agent_signals[agent_name] = signal
                self._agent_activation_count[agent_name] += 1
                self._agent_contribution_sum[agent_name] += abs(signal.action * signal.confidence)
        
        if not agent_signals:
            return 0.0, regime_name
        
        ensemble_output = self.ensemble.combine_signals(
            agent_signals, gating_output.weights, current_volatility=current_vol
        )
        
        controller_output = self.controller.compute_position(
            ensemble_output.final_action, ensemble_output.confidence,
            current_vol, regime_probs
        )
        
        if controller_output.is_trade_allowed:
            prev_pos = self.controller.current_position
            self.controller.execute_trade(controller_output.target_position)
            if 'returns' in df.columns and current_idx > 0:
                daily_ret = df['returns'].iloc[current_idx]
                if not pd.isna(daily_ret):
                    self.controller.update_equity(prev_pos * daily_ret)
        
        if self.debug and current_idx % 50 == 0:
            date_str = str(df.index[current_idx])[:10]
            active_str = ', '.join(gating_output.active_agents)
            
            print(f"    [{date_str}] Regime: {regime_name}, Pos: {controller_output.target_position:.2f}, "
                  f"Base: {controller_output.baseline:.2f}, Overlay: {controller_output.overlay:+.2f}")
            for name, sig in agent_signals.items():
                w = gating_output.weights.get(name, 0)
                print(f"      {name}: act={sig.action:.3f}, conf={sig.confidence:.2f}, w={w:.2f}")
        
        return controller_output.target_position, regime_name
    
    def get_agent_summary(self) -> str:
        lines = ["  Per-Agent Summary:"]
        for name in self.agents:
            count = self._agent_activation_count[name]
            avg = self._agent_contribution_sum[name] / max(1, count)
            lines.append(f"    {name}: active {count} bars, avg |act*conf|={avg:.3f}")
        return '\n'.join(lines)
    
    def reset(self):
        self.controller.reset()
        self.ensemble._action_history.clear()
        self.debug_log.clear()
        self._agent_activation_count = {n: 0 for n in self.agents}
        self._agent_contribution_sum = {n: 0.0 for n in self.agents}


def plot_results(ticker: str, result: BacktestResult, save_dir: str = '../Plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f'V4 MoA Strategy: {ticker}', fontsize=16, fontweight='bold')
    
    ax = axes[0]
    ax.plot(result.equity_curve.index, result.equity_curve.values, 
            label=f'MoA V4 ({result.total_return:.1%})', linewidth=2, color='#2196F3')
    ax.plot(result.benchmark_curve.index, result.benchmark_curve.values,
            label=f'Buy & Hold ({result.benchmark_return:.1%})', linewidth=1.5, 
            color='#FF9800', linestyle='--')
    ax.set_ylabel('Portfolio Value')
    ax.legend(loc='upper left')
    ax.set_title('Equity Curve')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.fill_between(result.drawdown_curve.index, 0, result.drawdown_curve.values,
                     color='#F44336', alpha=0.4)
    ax.set_ylabel('Drawdown')
    ax.set_title(f'Drawdown (Max: {result.max_drawdown:.1%})')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                     where=result.position_history >= 0, color='#4CAF50', alpha=0.5, label='Long')
    ax.fill_between(result.position_history.index, 0, result.position_history.values,
                     where=result.position_history < 0, color='#F44336', alpha=0.5, label='Short')
    ax.set_ylabel('Position')
    ax.set_title('Position History')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax = axes[3]
    regime_colors = {'Growth': '#4CAF50', 'Stagnation': '#9E9E9E', 
                     'Transition': '#FF9800', 'Crisis': '#F44336', 'Unknown': '#BDBDBD'}
    for rname, color in regime_colors.items():
        mask = result.regime_history == rname
        if mask.any():
            ax.fill_between(result.regime_history.index, 0, 1,
                           where=mask, color=color, alpha=0.4, label=rname)
    ax.set_ylabel('Regime')
    ax.set_title('Regime History')
    ax.legend(loc='upper right', ncol=4)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'v4_{ticker}_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(all_results: Dict[str, BacktestResult], save_dir: str = '../Plots'):
    os.makedirs(save_dir, exist_ok=True)
    tickers = list(all_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('V4 MoA Framework — Benchmark Summary', fontsize=16, fontweight='bold')
    
    x = np.arange(len(tickers))
    width = 0.35
    
    ax = axes[0]
    ax.bar(x - width/2, [all_results[t].total_return*100 for t in tickers], width, label='MoA V4', color='#2196F3')
    ax.bar(x + width/2, [all_results[t].benchmark_return*100 for t in tickers], width, label='Buy & Hold', color='#FF9800')
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel('Total Return (%)'); ax.set_title('Returns'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1]
    alphas = [all_results[t].alpha*100 for t in tickers]
    ax.bar(tickers, alphas, color=['#4CAF50' if a>=0 else '#F44336' for a in alphas])
    ax.set_ylabel('Alpha (%)'); ax.set_title('Alpha'); ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[2]
    sharpes = [all_results[t].sharpe_ratio for t in tickers]
    ax.bar(tickers, sharpes, color=['#4CAF50' if s>=0 else '#F44336' for s in sharpes])
    ax.set_ylabel('Sharpe'); ax.set_title('Sharpe Ratio'); ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'v4_benchmark_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def run_single_backtest(ticker: str, period: str = '2y', debug: bool = True) -> Optional[BacktestResult]:
    print(f"\n{'='*60}")
    print(f"Running V4 backtest for {ticker}")
    print(f"{'='*60}")
    
    loader = DataLoader()
    try:
        df = loader.fetch_ticker(ticker, period=period)
    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")
        return None
    
    df = add_all_indicators(df)
    strategy = MoAStrategyV4(debug=debug)
    strategy.regime_detector.fit(df)
    
    engine = BacktestEngine(transaction_cost=0.001)
    try:
        result = engine.run(df, strategy.generate_signal, warmup_period=60)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        return None
    
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
    print(f"    Avg |position|:     {pos.abs().mean():>.3f}")
    print(f"    Time near-zero:     {(pos.abs() < 0.1).mean():>.1%}")
    print(f"    Time long (>0.3):   {(pos > 0.3).mean():>.1%}")
    
    plot_results(ticker, result)
    return result


def run_full_benchmark(tickers=None, period='2y', debug=True):
    if tickers is None:
        tickers = get_benchmark_tickers()
    
    print(f"\n{'='*60}")
    print(f"MoA V4 Trading Framework — Full Benchmark")
    print(f"  Architecture: Adaptive Baseline + Alpha Overlay")
    print(f"{'='*60}")
    
    all_results = {}
    for ticker in tickers:
        result = run_single_backtest(ticker, period, debug)
        if result:
            all_results[ticker] = result
    
    if all_results:
        print(f"\n{'='*60}")
        print(f"V4 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Ticker':>6}  {'Return':>10}  {'B&H':>10}  {'Alpha':>8}  {'Sharpe':>8}  "
              f"{'MaxDD':>8}  {'Trades':>8}  {'AvgPos':>8}")
        
        for t, r in all_results.items():
            avg_pos = r.position_history.mean()
            print(f"{t:>6}  {r.total_return:>10.2%}  {r.benchmark_return:>10.2%}  "
                  f"{r.alpha:>8.2%}  {r.sharpe_ratio:>8.2f}  "
                  f"{r.max_drawdown:>8.2%}  {r.num_trades:>8d}  {avg_pos:>+8.3f}")
        
        avg_ret = np.mean([r.total_return for r in all_results.values()])
        avg_bh = np.mean([r.benchmark_return for r in all_results.values()])
        avg_alpha = np.mean([r.alpha for r in all_results.values()])
        avg_sharpe = np.mean([r.sharpe_ratio for r in all_results.values()])
        avg_dd = np.mean([r.max_drawdown for r in all_results.values()])
        
        print(f"\n  Avg Return:       {avg_ret:.2%}")
        print(f"  Avg Benchmark:    {avg_bh:.2%}")
        print(f"  Avg Alpha:        {avg_alpha:.2%}")
        print(f"  Avg Sharpe:       {avg_sharpe:.2f}")
        print(f"  Avg MaxDD:        {avg_dd:.2%}")
        print(f"  B&H Capture:      {avg_ret/avg_bh:.0%}")
        
        plot_summary(all_results)
        print(f"\n  Plots saved to Plots/")
    
    return all_results


if __name__ == '__main__':
    run_full_benchmark()
