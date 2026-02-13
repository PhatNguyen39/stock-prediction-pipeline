"""
Training script - Run complete training pipeline
"""
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.utils.logger import log
from src.utils.config import settings
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.data.features import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.models.trainer import ModelTrainer


def main(
    symbols: list = None,
    lookback_days: int = None,
    save_data: bool = True
):
    """
    Run the complete training pipeline
    
    Args:
        symbols: List of stock symbols (uses config if None)
        lookback_days: Days of historical data (uses config if None)
        save_data: Whether to save processed data
    """
    log.info("=" * 80)
    log.info("STOCK PREDICTION MODEL TRAINING")
    log.info("=" * 80)
    
    # Use config defaults if not provided
    if symbols is None:
        symbols = settings.symbol_list
    if lookback_days is None:
        lookback_days = settings.lookback_days
    
    log.info(f"Symbols: {symbols}")
    log.info(f"Lookback days: {lookback_days}")
    
    # Step 1: Fetch data
    log.info("\n[Step 1/5] Fetching stock data...")
    fetcher = DataFetcher(use_alpaca=False)  # Use Yahoo Finance
    df = fetcher.fetch(symbols=symbols)
    
    log.info(f"Fetched {len(df)} rows for {len(symbols)} symbols")
    
    # Step 2: Validate and clean data
    log.info("\n[Step 2/5] Validating and cleaning data...")
    validator = DataValidator()
    is_valid, metrics = validator.validate(df)
    
    if not is_valid:
        log.warning("Data validation found issues. Cleaning...")
    
    df_clean = validator.clean_data(df)
    log.info(f"Clean data: {len(df_clean)} rows")
    
    # Step 3: Feature engineering
    log.info("\n[Step 3/5] Engineering features...")
    feature_engineer = FeatureEngineer(window=settings.feature_window)
    df_features = feature_engineer.engineer_features(
        df_clean,
        create_target=True,
        target_horizon=settings.prediction_horizon
    )
    
    log.info(f"Features created: {df_features.shape}")
    
    # Save processed data
    if save_data:
        data_dir = Path("data/processed")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = data_dir / f"processed_data_{timestamp}.csv"
        df_features.to_csv(save_path, index=False)
        log.info(f"Processed data saved to {save_path}")
    
    # Step 4: Train model
    log.info("\n[Step 4/5] Training XGBoost model...")
    model = XGBoostModel(model_name="xgboost")
    trainer = ModelTrainer(mlflow_tracking=True)
    
    metrics = trainer.train_model(
        model=model,
        df=df_features,
        target_col='target_direction',
        train_size=0.7,
        val_size=0.15,
        log_mlflow=True,
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Step 5: Save model
    log.info("\n[Step 5/5] Saving model...")
    model_dir = Path("models/saved")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = model_dir / f"model_{timestamp}.pkl"
    model.save(str(versioned_path))
    log.info(f"Model saved to {versioned_path}")
    
    # Save as latest
    latest_path = model_dir / "latest_model.pkl"
    model.save(str(latest_path))
    log.info(f"Model saved as latest: {latest_path}")
    
    # Print feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        log.info("\nTop 15 Most Important Features:")
        log.info(importance.head(15).to_string(index=False))
    
    # Final summary
    log.info("\n" + "=" * 80)
    log.info("TRAINING COMPLETED SUCCESSFULLY")
    log.info("=" * 80)
    log.info(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
    log.info(f"Test AUC:       {metrics['test_auc']:.4f}")
    log.info(f"Test Precision: {metrics['test_precision']:.4f}")
    log.info(f"Test Recall:    {metrics['test_recall']:.4f}")
    log.info(f"Test F1:        {metrics['test_f1']:.4f}")
    log.info("=" * 80)
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Stock symbols to train on (e.g., AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Days of historical data to fetch"
    )
    parser.add_argument(
        "--no-save-data",
        action="store_true",
        help="Don't save processed data"
    )
    
    args = parser.parse_args()
    
    main(
        symbols=args.symbols,
        lookback_days=args.lookback_days,
        save_data=not args.no_save_data
    )
