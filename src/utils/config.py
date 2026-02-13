"""
Configuration management using Pydantic Settings
"""
from typing import List, Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    alpaca_api_key: str = Field(default="", description="Alpaca API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL"
    )
    
    # Model Configuration
    model_type: Literal["xgboost", "ensemble"] = Field(
        default="xgboost",
        description="Model type to use"
    )
    retrain_interval_hours: int = Field(
        default=24,
        description="Hours between model retraining"
    )
    prediction_horizon: int = Field(
        default=1,
        description="Days ahead to predict"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Enable hot reload")
    
    # Data Configuration
    symbols: str = Field(
        default="AAPL,MSFT,GOOGL,AMZN,TSLA",
        description="Comma-separated stock symbols"
    )
    lookback_days: int = Field(
        default=365,
        description="Days of historical data to fetch"
    )
    feature_window: int = Field(
        default=20,
        description="Window for technical indicators"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(
        default="./mlruns",
        description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="stock_prediction",
        description="MLflow experiment name"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="./logs/app.log", description="Log file path")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": (),
    }
    
    @property
    def symbol_list(self) -> List[str]:
        """Parse symbols into list"""
        return [s.strip() for s in self.symbols.split(",")]


# Global settings instance
settings = Settings()
