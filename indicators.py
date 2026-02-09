"""
Technical Indicators Module for MoA Trading Framework
Provides trend, mean-reversion, volatility, and momentum indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """
    Comprehensive technical indicators library.
    All methods are static and operate on pandas DataFrames/Series.
    """
    
    # ==================== TREND INDICATORS ====================
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(
        close: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def adx(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """
        Average Directional Index - measures trend strength.
        ADX > 25 indicates strong trend, < 20 indicates weak/no trend.
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    # ==================== MEAN REVERSION INDICATORS ====================
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        RSI > 70: overbought, RSI < 30: oversold.
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(
        close: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Returns:
            (upper_band, middle_band, lower_band)
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def bollinger_percent_b(
        close: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.Series:
        """
        Bollinger %B - position within the bands.
        > 1: above upper band, < 0: below lower band.
        """
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close, period, std_dev)
        percent_b = (close - lower) / (upper - lower + 1e-8)
        return percent_b
    
    @staticmethod
    def z_score(close: pd.Series, period: int = 20) -> pd.Series:
        """
        Z-Score - number of standard deviations from mean.
        """
        mean = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        return (close - mean) / (std + 1e-8)
    
    # ==================== VOLATILITY INDICATORS ====================
    
    @staticmethod
    def atr(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """Average True Range - volatility measure."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def historical_volatility(
        close: pd.Series, 
        period: int = 20, 
        annualize: bool = True
    ) -> pd.Series:
        """
        Historical volatility based on log returns.
        """
        log_returns = np.log(close / close.shift(1))
        vol = log_returns.rolling(window=period).std()
        if annualize:
            vol = vol * np.sqrt(252)
        return vol
    
    @staticmethod
    def volatility_ratio(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
        """
        Ratio of short-term to long-term volatility.
        > 1 indicates volatility expansion, < 1 indicates contraction.
        """
        short_vol = TechnicalIndicators.historical_volatility(close, short, annualize=False)
        long_vol = TechnicalIndicators.historical_volatility(close, long, annualize=False)
        return short_vol / (long_vol + 1e-8)
    
    @staticmethod
    def atr_percent(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """ATR as percentage of price."""
        atr = TechnicalIndicators.atr(high, low, close, period)
        return atr / close * 100
    
    # ==================== MOMENTUM INDICATORS ====================
    
    @staticmethod
    def roc(close: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change - momentum oscillator."""
        return (close - close.shift(period)) / close.shift(period) * 100
    
    @staticmethod
    def stochastic(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Returns:
            (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def momentum(close: pd.Series, period: int = 10) -> pd.Series:
        """Simple momentum - price difference over period."""
        return close - close.shift(period)
    
    # ==================== COMPOSITE INDICATORS ====================
    
    @staticmethod
    def trend_strength(
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> pd.Series:
        """
        Composite trend strength indicator.
        Combines ADX with price position relative to moving averages.
        Range: 0 to 100, higher = stronger trend.
        """
        adx = TechnicalIndicators.adx(high, low, close)
        sma_20 = TechnicalIndicators.sma(close, 20)
        sma_50 = TechnicalIndicators.sma(close, 50)
        
        # Price above both MAs = bullish, below both = bearish
        above_20 = (close > sma_20).astype(float)
        above_50 = (close > sma_50).astype(float)
        ma_alignment = (above_20 + above_50) / 2
        
        # Combine ADX with MA alignment
        trend_strength = adx * (0.5 + 0.5 * ma_alignment)
        return trend_strength.clip(0, 100)
    
    @staticmethod
    def mean_reversion_score(close: pd.Series) -> pd.Series:
        """
        Composite mean reversion score.
        High positive = overbought (sell), High negative = oversold (buy).
        Range: -100 to 100.
        """
        rsi = TechnicalIndicators.rsi(close)
        z_score = TechnicalIndicators.z_score(close)
        bb_pct = TechnicalIndicators.bollinger_percent_b(close)
        
        # Normalize RSI to -1 to 1
        rsi_norm = (rsi - 50) / 50
        
        # Normalize BB%B to -1 to 1
        bb_norm = (bb_pct - 0.5) * 2
        
        # Combine (z_score is already normalized around 0)
        score = (rsi_norm + bb_norm + z_score.clip(-3, 3) / 3) / 3 * 100
        return score.clip(-100, 100)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame with OHLCV data.
    """
    ti = TechnicalIndicators
    
    # Trend
    df['sma_20'] = ti.sma(df['close'], 20)
    df['sma_50'] = ti.sma(df['close'], 50)
    df['ema_12'] = ti.ema(df['close'], 12)
    df['ema_26'] = ti.ema(df['close'], 26)
    df['macd'], df['macd_signal'], df['macd_hist'] = ti.macd(df['close'])
    df['adx'] = ti.adx(df['high'], df['low'], df['close'])
    
    # Mean Reversion
    df['rsi'] = ti.rsi(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ti.bollinger_bands(df['close'])
    df['bb_pct_b'] = ti.bollinger_percent_b(df['close'])
    df['z_score'] = ti.z_score(df['close'])
    
    # Volatility
    df['atr'] = ti.atr(df['high'], df['low'], df['close'])
    df['atr_pct'] = ti.atr_percent(df['high'], df['low'], df['close'])
    df['hist_vol'] = ti.historical_volatility(df['close'])
    df['vol_ratio'] = ti.volatility_ratio(df['close'])
    
    # Momentum
    df['roc'] = ti.roc(df['close'])
    df['stoch_k'], df['stoch_d'] = ti.stochastic(df['high'], df['low'], df['close'])
    df['momentum'] = ti.momentum(df['close'])
    
    # Composite
    df['trend_strength'] = ti.trend_strength(df['high'], df['low'], df['close'])
    df['mr_score'] = ti.mean_reversion_score(df['close'])
    
    return df


if __name__ == "__main__":
    # Test the indicators
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.fetch_ticker('AAPL', period='3mo')
    df = add_all_indicators(df)
    
    print("Technical Indicators Test:")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLast 5 rows of key indicators:")
    print(df[['close', 'rsi', 'macd', 'adx', 'bb_pct_b', 'atr_pct', 'trend_strength', 'mr_score']].tail())
