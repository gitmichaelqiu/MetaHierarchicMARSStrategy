"""
Data Loader Module for MoA Trading Framework
Fetches OHLCV data from yfinance with rate-limit protection using curl_cffi
"""

import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests
from typing import Optional, List, Dict
from datetime import datetime, timedelta


class DataLoader:
    """
    Data loader with anti-rate-limit protection using curl_cffi sessions.
    Provides OHLCV data with calculated returns and rolling statistics.
    """
    
    def __init__(self):
        """Initialize with a curl_cffi session to bypass rate limits."""
        self.session = requests.Session(impersonate="chrome")
    
    def fetch_ticker(
        self, 
        ticker: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format  
            period: Alternative to date range (e.g., '1y', '2y', '5y')
            
        Returns:
            DataFrame with OHLCV + calculated fields
        """
        yf_ticker = yf.Ticker(ticker, session=self.session)
        
        if start_date and end_date:
            df = yf_ticker.history(start=start_date, end=end_date)
        else:
            df = yf_ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data fetched for {ticker}")
        
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Calculate additional fields
        df = self._add_calculated_fields(df)
        
        return df
    
    def fetch_multiple(
        self, 
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers.
        
        Returns:
            Dictionary mapping ticker -> DataFrame
        """
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = self.fetch_ticker(ticker, start_date, end_date, period)
                print(f"✓ Fetched {ticker}: {len(data[ticker])} rows")
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
        return data
    
    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields: returns, log returns, rolling stats."""
        
        # Daily returns
        df['returns'] = df['close'].pct_change()
        
        # Log returns (for statistical properties)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling statistics (20-day window)
        window = 20
        df['rolling_mean'] = df['close'].rolling(window=window).mean()
        df['rolling_std'] = df['returns'].rolling(window=window).std()
        df['rolling_volatility'] = df['rolling_std'] * np.sqrt(252)  # Annualized
        
        # Rolling skewness and kurtosis (for regime detection)
        df['rolling_skew'] = df['returns'].rolling(window=window).skew()
        df['rolling_kurt'] = df['returns'].rolling(window=window).kurt()
        
        # Range and body metrics
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['body'] = (df['close'] - df['open']) / df['open']
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        return df


def get_benchmark_tickers() -> List[str]:
    """Return the list of benchmark technology tickers."""
    return ['AAPL', 'MSFT', 'GOOG', 'NVDA', 'TSLA']


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    print("Testing DataLoader...")
    df = loader.fetch_ticker('AAPL', period='3mo')
    print(f"\nFetched AAPL data: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.tail())
