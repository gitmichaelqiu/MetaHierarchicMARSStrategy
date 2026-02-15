"""
CIO Visualization Module — Rich decision visualization.

Three main plots:
1. Per-Ticker Decision Strips — stock price with weight/regime shading
2. Portfolio Allocation Heatmap — time × ticker weight intensity
3. Portfolio Dashboard — equity, drawdown, cash, risk timeline
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


REGIME_COLORS = {
    'Growth': '#4CAF50',
    'Stagnation': '#FF9800',
    'Transition': '#9C27B0',
    'Crisis': '#F44336',
}

REGIME_ORDER = ['Growth', 'Stagnation', 'Transition', 'Crisis']


def plot_decision_strips(result, ticker_data, tickers, warmup=60,
                          allocator_name='CIO', save_dir='../Plots'):
    """
    Per-ticker decision strip: stock price with green shading = holding weight.
    Bottom strip shows regime classification.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n = len(tickers)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    fig.suptitle(f'CIO Decision Strips — {allocator_name}', fontsize=18, fontweight='bold', y=1.01)
    
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    wt_df = result.weight_history
    regime_df = result.regime_history
    
    for i, ticker in enumerate(tickers):
        ax = axes[i]
        
        # Get price data for overlapping period
        df = ticker_data[ticker]
        price = df['close']
        
        # Align dates
        common_dates = wt_df.index
        
        # Normalize price to start at 1.0 for the overlapping period
        price_aligned = price.reindex(common_dates, method='ffill')
        if price_aligned.notna().sum() == 0:
            ax.text(0.5, 0.5, f'{ticker}: No Data', ha='center', va='center')
            continue
        
        price_norm = price_aligned / price_aligned.dropna().iloc[0]
        
        # Plot price
        ax.plot(common_dates, price_norm, color='#1a1a2e', linewidth=1.2, zorder=3)
        
        # Green shading for holding periods — intensity = weight
        if ticker in wt_df.columns:
            weights = wt_df[ticker].values
            
            # Shade in blocks
            for j in range(len(common_dates) - 1):
                w = weights[j]
                if w > 0.01:  # Only shade if meaningful weight
                    alpha = min(0.7, w * 2.5)  # Scale: 10% weight → α=0.25, 30% → α=0.7
                    ax.axvspan(common_dates[j], common_dates[j + 1],
                              color='#2E7D32', alpha=alpha, linewidth=0)
        
        # Regime bar at bottom (thin strip)
        if ticker in regime_df.columns:
            regimes = regime_df[ticker]
            ax_ymin, ax_ymax = ax.get_ylim()
            bar_height = (ax_ymax - ax_ymin) * 0.04
            
            for j in range(len(common_dates) - 1):
                if pd.notna(regimes.iloc[j]):
                    regime = str(regimes.iloc[j])
                    color = REGIME_COLORS.get(regime, '#BDBDBD')
                    ax.axvspan(common_dates[j], common_dates[j + 1],
                              ymin=0, ymax=0.03, color=color, linewidth=0)
        
        # B&H return
        if price_norm.dropna().shape[0] > 1:
            bh_ret = price_norm.dropna().iloc[-1] - 1
            ax.set_title(f'{ticker}  (B&H: {bh_ret:+.1%})', fontsize=12, fontweight='bold')
        else:
            ax.set_title(ticker, fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Normalized Price')
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
    
    # Remove empty axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#2E7D32', alpha=0.4, label='Holding (weight)'),
        Patch(facecolor=REGIME_COLORS['Growth'], label='Growth'),
        Patch(facecolor=REGIME_COLORS['Stagnation'], label='Stagnation'),
        Patch(facecolor=REGIME_COLORS['Crisis'], label='Crisis'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=9,
              bbox_to_anchor=(0.98, 1.005))
    
    plt.tight_layout()
    fname = f'cio_decisions_{allocator_name.lower().replace(" ", "_").replace("-", "_")}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def plot_allocation_heatmap(result, tickers, allocator_name='CIO', save_dir='../Plots'):
    """
    Time × Ticker heatmap showing weight allocation over time.
    Rows = tickers, Columns = time. Color intensity = weight.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    wt_df = result.weight_history
    if wt_df.empty:
        return
    
    # Resample to weekly for readability
    wt_weekly = wt_df.resample('W').mean()
    
    # Build matrix: tickers × weeks
    matrix = np.zeros((len(tickers), len(wt_weekly)))
    for i, t in enumerate(tickers):
        if t in wt_weekly.columns:
            matrix[i] = wt_weekly[t].fillna(0).values
    
    fig, ax = plt.subplots(figsize=(20, max(5, len(tickers) * 0.6)))
    
    # Custom colormap: white → green
    cmap = LinearSegmentedColormap.from_list('weight', ['#FFFFFF', '#C8E6C9', '#2E7D32', '#1B5E20'])
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=0.35,
                    interpolation='nearest')
    
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=11, fontweight='bold')
    
    # X-axis: show monthly labels
    week_dates = wt_weekly.index
    n_weeks = len(week_dates)
    tick_positions = np.arange(0, n_weeks, max(1, n_weeks // 12))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([week_dates[p].strftime('%b %y') for p in tick_positions],
                       rotation=30, ha='right', fontsize=9)
    
    # Add cash row
    cash_weekly = result.cash_history.resample('W').mean().fillna(0).values
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Allocation Weight', fontsize=10)
    
    ax.set_title(f'Portfolio Allocation Heatmap — {allocator_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    fname = f'cio_heatmap_{allocator_name.lower().replace(" ", "_").replace("-", "_")}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def plot_dashboard(result, allocator_name='CIO', save_dir='../Plots'):
    """
    4-panel dashboard: equity curve, drawdown, weight allocation, risk timeline.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.5, 2.5, 1], hspace=0.35)
    
    # ── Panel 1: Equity Curve ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(result.equity_curve.index, result.equity_curve.values,
             linewidth=2.5, color='#1565C0', label=f'CIO Portfolio ({result.total_return:.1%})')
    ax1.axhline(y=1 + result.equal_weight_return, color='#FF6F00', linestyle='--',
                linewidth=1.5, label=f'Equal-Weight B&H ({result.equal_weight_return:.1%})')
    ax1.axhline(y=1.0, color='#BDBDBD', linestyle=':', linewidth=1)
    ax1.set_ylabel('Portfolio Value ×', fontsize=12)
    ax1.set_title(f'CIO Portfolio Dashboard — {allocator_name}', fontsize=18, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.fill_between(result.equity_curve.index,
                     result.equity_curve.values, 1.0,
                     where=result.equity_curve.values >= 1.0,
                     color='#1565C0', alpha=0.08)
    
    # ── Panel 2: Drawdown ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(result.drawdown_curve.index, 0,
                     -result.drawdown_curve.values * 100,
                     color='#C62828', alpha=0.5)
    ax2.set_ylabel('Drawdown %', fontsize=11)
    ax2.set_title(f'Drawdown (Max: {result.max_drawdown:.1%})', fontsize=12)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-result.max_drawdown * 120, 2)
    
    # ── Panel 3: Weight Stack ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    wt_df = result.weight_history
    if not wt_df.empty:
        # Sort tickers by total contribution
        ticker_order = sorted(wt_df.columns, key=lambda t: wt_df[t].sum(), reverse=True)
        
        palette = plt.cm.Set3(np.linspace(0, 1, len(ticker_order)))
        bottom = np.zeros(len(wt_df))
        for idx, t in enumerate(ticker_order):
            values = wt_df[t].fillna(0).values
            ax3.fill_between(wt_df.index, bottom, bottom + values,
                           color=palette[idx], alpha=0.85, label=t, linewidth=0)
            bottom += values
        
        # Cash on top
        ax3.fill_between(result.cash_history.index, bottom,
                        np.ones(len(result.cash_history)),
                        color='#E0E0E0', alpha=0.6, label='Cash', linewidth=0)
        
        ax3.set_ylabel('Allocation', fontsize=11)
        ax3.set_title(f'Weight Allocation (Avg Cash: {result.avg_cash_pct:.0%})', fontsize=12)
        ax3.legend(loc='upper right', ncol=6, fontsize=8, framealpha=0.9)
        ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.2)
    
    # ── Panel 4: Risk Flag Timeline ──
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    dates = result.equity_curve.index
    
    if hasattr(result, 'risk_flag_history') and result.risk_flag_history:
        flag_types = {
            'DD_': ('#F44336', 'Drawdown'),
            'CORR_': ('#FF9800', 'Correlation'),
            'VIX_': ('#9C27B0', 'VIX'),
            'CRISIS': ('#D32F2F', 'Crisis'),
        }
        
        y_pos = 0
        for prefix, (color, label) in flag_types.items():
            flags_active = []
            for flags in result.risk_flag_history:
                active = any(prefix in f for f in flags) if flags else False
                flags_active.append(active)
            
            # Pad if needed
            while len(flags_active) < len(dates):
                flags_active.append(False)
            flags_active = flags_active[:len(dates)]
            
            for j in range(len(dates) - 1):
                if flags_active[j]:
                    ax4.axvspan(dates[j], dates[j + 1],
                              ymin=y_pos / 4, ymax=(y_pos + 1) / 4,
                              color=color, alpha=0.7, linewidth=0)
            y_pos += 1
        
        ax4.set_yticks([0.125, 0.375, 0.625, 0.875])
        ax4.set_yticklabels(['DD', 'Corr', 'VIX', 'Crisis'], fontsize=9)
    
    ax4.set_title('Risk Flags Active', fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.2, axis='x')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    fname = f'cio_dashboard_{allocator_name.lower().replace(" ", "_").replace("-", "_")}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fname}")


def plot_all_visualizations(result, ticker_data, tickers, allocator_name='CIO',
                             save_dir='../Plots'):
    """Generate all three visualization types."""
    print(f"\n  Generating visualizations for {allocator_name}...")
    plot_decision_strips(result, ticker_data, tickers,
                          allocator_name=allocator_name, save_dir=save_dir)
    plot_allocation_heatmap(result, tickers,
                             allocator_name=allocator_name, save_dir=save_dir)
    plot_dashboard(result, allocator_name=allocator_name, save_dir=save_dir)
