"""
Data fetching from Yahoo Finance and Alpaca
"""
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.utils.logger import log
from src.utils.config import settings


class DataFetcher:
    """Fetch stock data from multiple sources"""
    
    def __init__(self, use_alpaca: bool = False):
        """
        Initialize data fetcher
        
        Args:
            use_alpaca: Whether to use Alpaca API (requires API keys)
        """
        self.use_alpaca = use_alpaca
        self.alpaca_client = None
        
        if use_alpaca and settings.alpaca_api_key:
            try:
                self.alpaca_client = StockHistoricalDataClient(
                    api_key=settings.alpaca_api_key,
                    secret_key=settings.alpaca_secret_key
                )
                log.info("Alpaca client initialized successfully")
            except Exception as e:
                log.warning(f"Failed to initialize Alpaca client: {e}")
                self.alpaca_client = None
    
    def fetch_yahoo_finance(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=settings.lookback_days)
        if end_date is None:
            end_date = datetime.now()
        
        log.info(f"Fetching data from Yahoo Finance for {symbols}")
        
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d")
                )
                
                if df.empty:
                    log.warning(f"No data found for {symbol}")
                    continue
                
                # Reset index to make Date a column
                df = df.reset_index()
                df['symbol'] = symbol
                
                # Rename columns to lowercase
                df.columns = df.columns.str.lower()
                
                # Select relevant columns
                df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                
                all_data.append(df)
                log.info(f"Fetched {len(df)} rows for {symbol}")
                
            except Exception as e:
                log.error(f"Error fetching {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data fetched for any symbol")
        
        # Combine all symbols
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        log.info(f"Total rows fetched: {len(combined_df)}")
        return combined_df
    
    def fetch_alpaca(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch data from Alpaca API
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.alpaca_client is None:
            raise ValueError("Alpaca client not initialized")
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=settings.lookback_days)
        if end_date is None:
            end_date = datetime.now()
        
        log.info(f"Fetching data from Alpaca for {symbols}")
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = self.alpaca_client.get_stock_bars(request_params)
        df = bars.df
        
        # Reset multi-index
        df = df.reset_index()
        
        # Rename columns
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'timestamp': 'date'})
        
        log.info(f"Fetched {len(df)} rows from Alpaca")
        return df
    
    def fetch(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch data using configured source
        
        Args:
            symbols: List of stock symbols (uses config if not provided)
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        if symbols is None:
            symbols = settings.symbol_list
        
        if self.use_alpaca and self.alpaca_client:
            return self.fetch_alpaca(symbols, start_date, end_date)
        else:
            return self.fetch_yahoo_finance(symbols, start_date, end_date)
