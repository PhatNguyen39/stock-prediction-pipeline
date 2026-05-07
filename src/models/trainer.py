"""
Model training with proper time-series validation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import mlflow.sklearn
from src.models.base import BaseModel
from src.utils.logger import log
from src.utils.config import settings


class ModelTrainer:
    """Train models with time-series aware validation"""
    
    def __init__(self, mlflow_tracking: bool = True):
        """
        Initialize trainer
        
        Args:
            mlflow_tracking: Whether to log to MLflow
        """
        self.mlflow_tracking = mlflow_tracking
        
        if mlflow_tracking:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
    
    def time_series_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically for time series
        
        Args:
            df: DataFrame with 'date' column
            train_size: Proportion for training
            val_size: Proportion for validation
            
        Returns:
            (train_df, val_df, test_df)
        """
        df = df.sort_values('date').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        log.info(f"Time series split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        log.info(f"Train period: {train_df['date'].min()} to {train_df['date'].max()}")
        log.info(f"Val period: {val_df['date'].min()} to {val_df['date'].max()}")
        log.info(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
        
        return train_df, val_df, test_df
    
    def prepare_features_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_direction',
        exclude_cols: list = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            exclude_cols: Additional columns to exclude
            
        Returns:
            (X, y)
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Default columns to exclude
        default_exclude = [
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'future_close', 'target_return', 'target_direction'
        ]
        
        all_exclude = list(set(default_exclude + exclude_cols))
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in all_exclude]
        
        X = df[feature_cols]
        y = df[target_col]
        
        log.info(f"Features: {len(feature_cols)} columns")
        log.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC if probabilities provided
        if y_proba is not None:
            metrics[f'{prefix}auc'] = roc_auc_score(y_true, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics[f'{prefix}true_negatives'] = int(tn)
            metrics[f'{prefix}false_positives'] = int(fp)
            metrics[f'{prefix}false_negatives'] = int(fn)
            metrics[f'{prefix}true_positives'] = int(tp)
        
        return metrics
    
    def train_model(
        self,
        model: BaseModel,
        df: pd.DataFrame,
        target_col: str = 'target_direction',
        train_size: float = 0.7,
        val_size: float = 0.15,
        log_mlflow: bool = True,
        **train_kwargs
    ) -> Dict[str, Any]:
        """
        Train model with proper validation
        
        Args:
            model: Model instance to train
            df: DataFrame with features and target
            target_col: Target column name
            train_size: Training set proportion
            val_size: Validation set proportion
            log_mlflow: Whether to log to MLflow
            **train_kwargs: Additional arguments for model.train()
            
        Returns:
            Dictionary with all metrics
        """
        log.info(f"Starting training for {model.model_name}")
        
        # Time series split
        train_df, val_df, test_df = self.time_series_split(df, train_size, val_size)
        
        # Prepare features and target
        X_train, y_train = self.prepare_features_target(train_df, target_col)
        X_val, y_val = self.prepare_features_target(val_df, target_col)
        X_test, y_test = self.prepare_features_target(test_df, target_col)
        
        # Start MLflow run
        if log_mlflow and self.mlflow_tracking:
            mlflow.start_run(run_name=f"{model.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Log parameters
            mlflow.log_param("model_type", model.model_name)
            mlflow.log_param("train_size", len(train_df))
            mlflow.log_param("val_size", len(val_df))
            mlflow.log_param("test_size", len(test_df))
            mlflow.log_param("n_features", X_train.shape[1])
        
        # Train model
        train_metrics = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **train_kwargs
        )
        
        # Evaluate on all sets
        all_metrics = {}
        
        # Training set
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_eval = self.calculate_metrics(y_train, y_train_pred, y_train_proba, prefix="train_")
        all_metrics.update(train_eval)
        
        # Validation set
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_eval = self.calculate_metrics(y_val, y_val_pred, y_val_proba, prefix="val_")
        all_metrics.update(val_eval)
        
        # Test set
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_eval = self.calculate_metrics(y_test, y_test_pred, y_test_proba, prefix="test_")
        all_metrics.update(test_eval)
        
        # Log metrics to MLflow
        if log_mlflow and self.mlflow_tracking:
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log feature importance if available
            importance = model.get_feature_importance()
            if importance is not None:
                importance_path = "feature_importance.csv"
                importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            mlflow.end_run()
        
        # Print summary
        log.info("=" * 60)
        log.info(f"Training completed for {model.model_name}")
        log.info(f"Test Accuracy: {all_metrics['test_accuracy']:.4f}")
        log.info(f"Test AUC: {all_metrics['test_auc']:.4f}")
        log.info(f"Test Precision: {all_metrics['test_precision']:.4f}")
        log.info(f"Test Recall: {all_metrics['test_recall']:.4f}")
        log.info(f"Test F1: {all_metrics['test_f1']:.4f}")
        log.info("=" * 60)
        
        return all_metrics

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        model_params: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        gap: int = None,
        target_col: str = 'target_direction',
        early_stopping_rounds: int = 30,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Expanding-window walk-forward cross-validation.

        Each fold grows the training window and tests on the next unseen period.
        A gap equal to prediction_horizon is inserted between train and test to
        avoid target-leakage from lookahead labels at the boundary.

        Args:
            df: Full feature-engineered DataFrame with date column.
            model_params: XGBoost params dict (uses model defaults if None).
            n_splits: Number of folds (default 5).
            gap: Rows to skip between train end and test start (defaults to
                 settings.prediction_horizon).
            target_col: Name of the target column.
            early_stopping_rounds: Early stopping patience per fold.

        Returns:
            (fold_metrics, avg_metrics) — per-fold dicts and cross-fold averages.
        """
        from src.models.xgboost_model import XGBoostModel

        if gap is None:
            gap = settings.prediction_horizon

        df = df.sort_values('date').reset_index(drop=True)
        X, y = self.prepare_features_target(df, target_col)
        dates = df['date']

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

        fold_metrics: List[Dict] = []

        log.info(f"Walk-forward validation: {n_splits} folds, gap={gap} rows")
        log.info(f"{'Fold':<6} {'Train':>8} {'Test':>8} {'Accuracy':>10} {'Precision':>10} {'AUC':>8}")
        log.info("-" * 55)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            # Internal val set = last 15% of training for early stopping
            val_split = int(len(train_idx) * 0.85)
            internal_train_idx = train_idx[:val_split]
            internal_val_idx   = train_idx[val_split:]

            X_train = X.iloc[internal_train_idx]
            y_train = y.iloc[internal_train_idx]
            X_val   = X.iloc[internal_val_idx]
            y_val   = y.iloc[internal_val_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]

            train_period = f"{dates.iloc[train_idx[0]].date()} → {dates.iloc[train_idx[-1]].date()}"
            test_period  = f"{dates.iloc[test_idx[0]].date()} → {dates.iloc[test_idx[-1]].date()}"

            model = XGBoostModel(params=model_params)
            model.train(X_train, y_train, X_val, y_val,
                        early_stopping_rounds=early_stopping_rounds, verbose=False)

            metrics = self.calculate_metrics(
                y_test, model.predict(X_test),
                model.predict_proba(X_test)[:, 1], prefix=""
            )
            metrics['fold']         = fold
            metrics['train_size']   = len(train_idx)
            metrics['test_size']    = len(test_idx)
            metrics['train_period'] = train_period
            metrics['test_period']  = test_period

            fold_metrics.append(metrics)

            log.info(
                f"{fold:<6} {len(train_idx):>8} {len(test_idx):>8}"
                f" {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f}"
                f" {metrics['auc']:>8.4f}"
            )

        # Average and std across folds
        avg_metrics: Dict[str, float] = {}
        log.info("-" * 55)
        log.info(f"{'Metric':<20} {'Mean':>8} {'Std':>8}")
        log.info("-" * 40)
        for key in ['accuracy', 'precision', 'auc']:
            values = [m[key] for m in fold_metrics]
            avg_metrics[f'avg_{key}'] = float(np.mean(values))
            avg_metrics[f'std_{key}'] = float(np.std(values))
            log.info(
                f"{key.capitalize():<20}"
                f" {avg_metrics[f'avg_{key}']:>8.4f}"
                f" ± {avg_metrics[f'std_{key}']:>6.4f}"
            )

        return fold_metrics, avg_metrics
