"""
Data validation and quality checks
"""
import pandas as pd
from typing import Dict, List, Tuple
from src.utils.logger import log


class DataValidator:
    """Validate data quality and schema"""
    
    def __init__(self):
        self.required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame has required columns
        
        Returns:
            (is_valid, missing_columns)
        """
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        
        if missing_cols:
            log.error(f"Missing required columns: {missing_cols}")
            return False, missing_cols
        
        return True, []
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check data quality metrics
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Check for missing values
        metrics['missing_values'] = df[self.required_columns].isnull().sum().to_dict()
        metrics['total_missing'] = df[self.required_columns].isnull().sum().sum()
        
        # Check for duplicates
        metrics['duplicate_rows'] = df.duplicated(subset=['date', 'symbol']).sum()
        
        # Check for price anomalies
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            metrics[f'{col}_negative'] = (df[col] <= 0).sum()
            metrics[f'{col}_null'] = df[col].isnull().sum()
        
        # Check volume
        metrics['volume_zero'] = (df['volume'] == 0).sum()
        metrics['volume_negative'] = (df['volume'] < 0).sum()
        
        # Check high >= low
        metrics['invalid_high_low'] = (df['high'] < df['low']).sum()
        
        # Check for extreme price changes (>50% in one day)
        df_sorted = df.sort_values(['symbol', 'date'])
        df_sorted['price_change'] = df_sorted.groupby('symbol')['close'].pct_change().abs()
        metrics['extreme_changes'] = (df_sorted['price_change'] > 0.5).sum()
        
        # Log warnings
        issues = []
        if metrics['total_missing'] > 0:
            issues.append(f"Found {metrics['total_missing']} missing values")
        if metrics['duplicate_rows'] > 0:
            issues.append(f"Found {metrics['duplicate_rows']} duplicate rows")
        if metrics['invalid_high_low'] > 0:
            issues.append(f"Found {metrics['invalid_high_low']} rows where high < low")
        
        if issues:
            log.warning("Data quality issues found:")
            for issue in issues:
                log.warning(f"  - {issue}")
        else:
            log.info("Data quality validation passed")
        
        return metrics
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Run all validations
        
        Returns:
            (is_valid, metrics)
        """
        # Schema validation
        schema_valid, missing_cols = self.validate_schema(df)
        if not schema_valid:
            return False, {'error': f'Missing columns: {missing_cols}'}
        
        # Quality validation
        metrics = self.validate_data_quality(df)
        
        # Determine if data is valid
        is_valid = (
            metrics['total_missing'] == 0 and
            metrics['duplicate_rows'] == 0 and
            metrics['invalid_high_low'] == 0
        )
        
        return is_valid, metrics
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean common data issues
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
        
        # Remove rows with negative or zero prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df = df[df[col] > 0]
        
        # Remove rows with negative volume
        df = df[df['volume'] >= 0]
        
        # Remove rows where high < low
        df = df[df['high'] >= df['low']]
        
        # Remove extreme outliers (>100% change in one day)
        df = df.sort_values(['symbol', 'date'])
        df['price_change'] = df.groupby('symbol')['close'].pct_change().abs()
        df = df[df['price_change'] <= 1.0]
        df = df.drop(columns=['price_change'])
        
        cleaned_rows = len(df)
        removed_rows = initial_rows - cleaned_rows
        
        if removed_rows > 0:
            log.info(f"Data cleaning removed {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df
