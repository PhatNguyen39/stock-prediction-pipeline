"""
Model training with proper time-series validation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
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
