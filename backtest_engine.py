"""
Backtest Engine for MoA Trading Framework
Event-driven backtesting with transaction costs and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeRecord:
    """Record of a single trade."""
    date: datetime
    action: str  # 'BUY', 'SELL', 'CLOSE'
    position_before: float
    position_after: float
    price: float
    transaction_cost: float
    

@dataclass
class BacktestResult:
    """
    Results from a backtest run.
    
    Contains performance metrics and history.
    """
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Comparison
    benchmark_return: float
    alpha: float
    beta: float
    
    # Trade statistics
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)
    position_history: pd.Series = field(default_factory=pd.Series)
    regime_history: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Backtest Results Summary
========================
Total Return:      {self.total_return:>8.2%}
Annualized Return: {self.annualized_return:>8.2%}
Benchmark Return:  {self.benchmark_return:>8.2%}
Alpha:             {self.alpha:>8.2%}

Sharpe Ratio:      {self.sharpe_ratio:>8.2f}
Sortino Ratio:     {self.sortino_ratio:>8.2f}
Max Drawdown:      {self.max_drawdown:>8.2%}
Calmar Ratio:      {self.calmar_ratio:>8.2f}

Num Trades:        {self.num_trades:>8d}
Win Rate:          {self.win_rate:>8.2%}
Profit Factor:     {self.profit_factor:>8.2f}
"""


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Features:
    - Transaction cost modeling
    - Position tracking
    - Performance metrics calculation
    - Regime tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.04,  # 4% annual
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction
            slippage: Slippage as fraction of price
            risk_free_rate: Annual risk-free rate for Sharpe
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        # State
        self._reset()
    
    def _reset(self):
        """Reset engine state."""
        self.capital = self.initial_capital
        self.position = 0.0  # Position in units of notional
        self.position_pct = 0.0  # Position as % of capital
        
        self.equity_history: List[float] = []
        self.position_history: List[float] = []
        self.regime_history: List[str] = []
        self.dates: List[datetime] = []
        self.trades: List[TradeRecord] = []
    
    def _execute_trade(
        self, 
        target_position_pct: float, 
        current_price: float,
        date: datetime
    ) -> float:
        """
        Execute a trade to reach target position.
        
        Returns:
            Transaction cost incurred
        """
        position_change = target_position_pct - self.position_pct
        
        if abs(position_change) < 0.01:  # Ignore tiny changes
            return 0.0
        
        # Calculate notional change
        notional_change = abs(position_change * self.capital)
        
        # Apply slippage (worse price for us)
        if position_change > 0:  # Buying
            effective_price = current_price * (1 + self.slippage)
        else:  # Selling
            effective_price = current_price * (1 - self.slippage)
        
        # Calculate transaction cost
        cost = notional_change * self.transaction_cost
        
        # Record trade
        action = 'BUY' if position_change > 0 else 'SELL'
        if abs(target_position_pct) < 0.01:
            action = 'CLOSE'
        
        self.trades.append(TradeRecord(
            date=date,
            action=action,
            position_before=self.position_pct,
            position_after=target_position_pct,
            price=current_price,
            transaction_cost=cost
        ))
        
        # Update position and capital
        self.position_pct = target_position_pct
        self.capital -= cost
        
        return cost
    
    def _calculate_pnl(
        self, 
        prev_price: float, 
        current_price: float
    ) -> float:
        """Calculate P&L from price change."""
        if abs(self.position_pct) < 0.01:
            return 0.0
        
        price_return = (current_price - prev_price) / prev_price
        pnl = self.capital * self.position_pct * price_return
        return pnl
    
    def run(
        self,
        df: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame, int], Tuple[float, str]],
        warmup_period: int = 50
    ) -> BacktestResult:
        """
        Run backtest on data.
        
        Args:
            df: DataFrame with OHLCV and indicators
            signal_func: Function(df, current_idx) -> (position_pct, regime)
            warmup_period: Bars to skip for indicator warmup
            
        Returns:
            BacktestResult with performance metrics
        """
        self._reset()
        
        if len(df) <= warmup_period:
            raise ValueError("Not enough data for backtest")
        
        # Initialize
        prev_price = df['close'].iloc[warmup_period]
        
        for i in range(warmup_period, len(df)):
            current_date = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Get signal
            target_position, regime = signal_func(df, i)
            
            # Execute trade if needed
            cost = self._execute_trade(target_position, current_price, current_date)
            
            # Calculate P&L
            pnl = self._calculate_pnl(prev_price, current_price)
            self.capital += pnl
            
            # Record state
            self.equity_history.append(self.capital)
            self.position_history.append(self.position_pct)
            self.regime_history.append(regime)
            self.dates.append(current_date)
            
            prev_price = current_price
        
        # Calculate metrics
        return self._calculate_metrics(df, warmup_period)
    
    def _calculate_metrics(
        self, 
        df: pd.DataFrame, 
        warmup_period: int
    ) -> BacktestResult:
        """Calculate performance metrics."""
        
        # Create series
        equity = pd.Series(self.equity_history, index=self.dates)
        positions = pd.Series(self.position_history, index=self.dates)
        regimes = pd.Series(self.regime_history, index=self.dates)
        
        # Returns
        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / self.initial_capital) - 1
        
        # Annualized return
        days = (self.dates[-1] - self.dates[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        if volatility > 0:
            sharpe = (excess_returns.mean() * 252) / volatility
        else:
            sharpe = 0.0
        
        # Sortino Ratio
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else volatility
        if downside_vol > 0:
            sortino = (excess_returns.mean() * 252) / downside_vol
        else:
            sortino = 0.0
        
        # Max Drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calmar Ratio
        if max_drawdown > 0:
            calmar = annualized_return / max_drawdown
        else:
            calmar = 0.0
        
        # Benchmark (buy and hold)
        benchmark_prices = df['close'].iloc[warmup_period:]
        benchmark_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
        benchmark_curve = (benchmark_prices / benchmark_prices.iloc[0]) * self.initial_capital
        benchmark_curve.index = self.dates
        
        # Alpha and Beta
        benchmark_returns = benchmark_prices.pct_change().dropna()
        if len(benchmark_returns) > 0 and len(returns) > 0:
            # Align indices
            common_dates = returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 10:
                r = returns.loc[common_dates]
                b = benchmark_returns.loc[common_dates]
                
                cov = np.cov(r, b)
                if cov[1, 1] > 0:
                    beta = cov[0, 1] / cov[1, 1]
                    alpha = (r.mean() - beta * b.mean()) * 252
                else:
                    beta = 1.0
                    alpha = 0.0
            else:
                beta = 1.0
                alpha = total_return - benchmark_return
        else:
            beta = 1.0
            alpha = 0.0
        
        # Trade statistics
        num_trades = len(self.trades)
        
        if num_trades > 0:
            # Calculate trade P&L
            trade_pnls = []
            for trade in self.trades:
                # Simple approximation
                trade_pnls.append(trade.position_after - trade.position_before)
            
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]
            
            win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity,
            benchmark_curve=benchmark_curve,
            position_history=positions,
            regime_history=regimes,
            drawdown_curve=drawdown
        )


if __name__ == "__main__":
    # Simple test
    from data_loader import DataLoader
    from indicators import add_all_indicators
    
    loader = DataLoader()
    df = loader.fetch_ticker('AAPL', period='1y')
    df = add_all_indicators(df)
    
    # Simple signal function for testing
    def simple_signal(df, idx):
        if df['rsi'].iloc[idx] > 70:
            return -0.5, 'Overbought'
        elif df['rsi'].iloc[idx] < 30:
            return 0.5, 'Oversold'
        return 0.0, 'Neutral'
    
    engine = BacktestEngine()
    result = engine.run(df, simple_signal)
    
    print(result.summary())
