"""
Feature engineering for stock prediction
"""
import pandas as pd
import numpy as np
from bisect import bisect_right
from typing import List
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from src.utils.logger import log
from src.utils.config import settings


class FeatureEngineer:
    """Create features for stock prediction"""
    
    def __init__(self, window: int = 20):
        """
        Initialize feature engineer
        
        Args:
            window: Window size for technical indicators
        """
        self.window = window
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Create target variable (future returns)
        
        Args:
            df: DataFrame with stock data
            horizon: Days ahead to predict
            
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        
        # Group by symbol to avoid mixing different stocks
        df['future_close'] = df.groupby('symbol')['close'].shift(-horizon)
        df['target_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Binary classification: price goes up (1) or down (0)
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        df = df.copy()
        
        # Returns
        df['return_1d'] = df.groupby('symbol')['close'].pct_change(1)
        df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
        df['return_10d'] = df.groupby('symbol')['close'].pct_change(10)
        df['return_20d'] = df.groupby('symbol')['close'].pct_change(20)
        
        # Price momentum
        df['price_momentum'] = df['close'] / df.groupby('symbol')['close'].shift(self.window) - 1
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Close position in day's range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['close_position'] = df['close_position'].fillna(0.5)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df = df.copy()
        
        # Volume changes
        df['volume_change'] = df.groupby('symbol')['volume'].pct_change(1)
        df['volume_ma_ratio'] = df['volume'] / df.groupby('symbol')['volume'].rolling(self.window).mean().reset_index(0, drop=True)
        
        # Volume-price trend
        price_change = df.groupby('symbol')['close'].pct_change()
        df['vpt'] = (df['volume'] * price_change).groupby(df['symbol']).cumsum()
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        df = df.copy()
        
        # Process each symbol separately
        all_symbols = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Moving Averages
            sma = SMAIndicator(close=symbol_df['close'], window=self.window)
            symbol_df['sma'] = sma.sma_indicator()
            
            ema = EMAIndicator(close=symbol_df['close'], window=self.window)
            symbol_df['ema'] = ema.ema_indicator()
            
            # MACD
            macd = MACD(close=symbol_df['close'])
            symbol_df['macd'] = macd.macd()
            symbol_df['macd_signal'] = macd.macd_signal()
            symbol_df['macd_diff'] = macd.macd_diff()
            
            # RSI
            rsi = RSIIndicator(close=symbol_df['close'], window=14)
            symbol_df['rsi'] = rsi.rsi()
            
            # Bollinger Bands
            bollinger = BollingerBands(close=symbol_df['close'], window=self.window)
            symbol_df['bb_high'] = bollinger.bollinger_hband()
            symbol_df['bb_low'] = bollinger.bollinger_lband()
            symbol_df['bb_mid'] = bollinger.bollinger_mavg()
            symbol_df['bb_width'] = (symbol_df['bb_high'] - symbol_df['bb_low']) / symbol_df['bb_mid']
            symbol_df['bb_position'] = (symbol_df['close'] - symbol_df['bb_low']) / (symbol_df['bb_high'] - symbol_df['bb_low'])
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(
                high=symbol_df['high'],
                low=symbol_df['low'],
                close=symbol_df['close'],
                window=14
            )
            symbol_df['stoch_k'] = stoch.stoch()
            symbol_df['stoch_d'] = stoch.stoch_signal()
            
            # ATR (Average True Range)
            atr = AverageTrueRange(
                high=symbol_df['high'],
                low=symbol_df['low'],
                close=symbol_df['close'],
                window=14
            )
            symbol_df['atr'] = atr.average_true_range()
            symbol_df['atr_ratio'] = symbol_df['atr'] / symbol_df['close']
            
            all_symbols.append(symbol_df)
        
        df = pd.concat(all_symbols, ignore_index=True)
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features"""
        df = df.copy()
        
        for lag in lags:
            df[f'close_lag_{lag}'] = df.groupby('symbol')['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df.groupby('symbol')['volume'].shift(lag)
            df[f'return_lag_{lag}'] = df.groupby('symbol')['return_1d'].shift(lag)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding for day of week
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch and merge external market features:
          - VIX: market fear/volatility regime
          - SPY 1d/5d return: broad market momentum
          - XLK 1d/5d return: tech sector momentum
          - days_to_earnings: proximity to next earnings announcement
        """
        import yfinance as yf

        def _naive_date(s: pd.Series) -> pd.Series:
            """Strip timezone and normalize to midnight — works on any datetime Series."""
            s = pd.to_datetime(s)
            if s.dt.tz is not None:
                s = s.dt.tz_convert('UTC').dt.tz_localize(None)
            return s.dt.normalize()

        df = df.copy()
        df['date'] = _naive_date(df['date'])

        start = df['date'].min() - pd.Timedelta(days=30)  # buffer for pct_change warmup
        end   = df['date'].max() + pd.Timedelta(days=1)

        # --- VIX, SPY, XLK ---
        raw = {}
        for ticker in ['^VIX', 'SPY', 'XLK']:
            try:
                data = yf.download(ticker, start=start, end=end,
                                   progress=False, auto_adjust=True)
                close = data['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                raw[ticker] = close.rename(ticker)
                log.info(f"Fetched {ticker}: {len(data)} rows")
            except Exception as e:
                log.warning(f"Could not fetch {ticker}: {e}")

        if raw:
            ext = pd.concat(raw.values(), axis=1)
            ext.index.name = 'date'
            ext = ext.reset_index()
            ext['date'] = _naive_date(ext['date'])

            if '^VIX' in ext.columns:
                ext['vix'] = ext['^VIX']
            if 'SPY' in ext.columns:
                ext['spy_return_1d'] = ext['SPY'].pct_change(1)
                ext['spy_return_5d'] = ext['SPY'].pct_change(5)
            if 'XLK' in ext.columns:
                ext['xlk_return_1d'] = ext['XLK'].pct_change(1)
                ext['xlk_return_5d'] = ext['XLK'].pct_change(5)

            keep = ['date', 'vix', 'spy_return_1d', 'spy_return_5d',
                    'xlk_return_1d', 'xlk_return_5d']
            keep = [c for c in keep if c in ext.columns]
            df = df.merge(ext[keep], on='date', how='left')
            log.info(f"Merged market features: {[c for c in keep if c != 'date']}")

        # --- Days to next earnings ---
        earnings_lookup = {}
        for symbol in df['symbol'].unique():
            try:
                ed = yf.Ticker(symbol).get_earnings_dates(limit=40)
                if ed is not None and not ed.empty:
                    idx = pd.to_datetime(ed.index)
                    if idx.tz is not None:
                        idx = idx.tz_convert('UTC').tz_localize(None)
                    dates = sorted(idx.normalize().tolist())
                    earnings_lookup[symbol] = dates
                    log.info(f"Fetched {len(dates)} earnings dates for {symbol}")
            except Exception as e:
                log.warning(f"Could not fetch earnings dates for {symbol}: {e}")

        if earnings_lookup:
            def _days_to_next(row):
                dates = earnings_lookup.get(row['symbol'], [])
                if not dates:
                    return 90
                idx = bisect_right(dates, row['date'])
                if idx >= len(dates):
                    return 90
                return min((dates[idx] - row['date']).days, 90)

            df['days_to_earnings'] = df.apply(_days_to_next, axis=1)
            log.info("Added days_to_earnings feature")

        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        create_target: bool = True,
        target_horizon: int = 1
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw DataFrame with OHLCV data
            create_target: Whether to create target variable
            target_horizon: Days ahead to predict
            
        Returns:
            DataFrame with engineered features
        """
        log.info("Starting feature engineering")
        
        df = df.copy()
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Create target first (needs to be before adding features)
        if create_target:
            df = self.create_target(df, horizon=target_horizon)
        
        # Add all features
        df = self.add_price_features(df)
        df = self.add_volume_features(df)
        df = self.add_technical_indicators(df)
        df = self.add_lag_features(df)
        df = self.add_time_features(df)
        df = self.add_external_features(df)
        
        # Drop rows with NaN (from rolling windows and lags)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        log.info(f"Feature engineering complete. Dropped {dropped_rows} rows with NaN")
        log.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names (excluding metadata and target)"""
        exclude_cols = [
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'future_close', 'target_return', 'target_direction'
        ]
        
        # This will be populated after engineering features once
        # For now, return common feature patterns
        return None  # Will be determined from actual DataFrame
