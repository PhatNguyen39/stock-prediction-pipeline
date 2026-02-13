"""
XGBoost model implementation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from src.models.base import BaseModel
from src.utils.logger import log


class XGBoostModel(BaseModel):
    """XGBoost classifier for stock prediction"""
    
    def __init__(
        self,
        model_name: str = "xgboost",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize XGBoost model
        
        Args:
            model_name: Model identifier
            params: XGBoost parameters (uses defaults if None)
        """
        super().__init__(model_name)
        
        # Default parameters optimized for binary classification
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'tree_method': 'hist',  # Fast histogram-based algorithm
            'enable_categorical': False
        }
        
        # Update with user-provided params
        if params:
            self.default_params.update(params)
        
        self.model = xgb.XGBClassifier(**self.default_params)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training metrics
        """
        log.info(f"Training {self.model_name} model")
        log.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        self.feature_names = list(X_train.columns)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            log.info(f"Validation samples: {len(X_val)}")
        
        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Calculate metrics
        metrics = {}
        
        # Training metrics
        y_train_pred = self.predict(X_train)
        y_train_proba = self.predict_proba(X_train)[:, 1]
        
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train_auc'] = roc_auc_score(y_train, y_train_proba)
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            y_val_proba = self.predict_proba(X_val)[:, 1]
            
            metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
            metrics['val_auc'] = roc_auc_score(y_val, y_val_proba)
            
            log.info(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
            log.info(f"Validation AUC: {metrics['val_auc']:.4f}")
        
        # Best iteration
        if hasattr(self.model, 'best_iteration'):
            metrics['best_iteration'] = self.model.best_iteration
            log.info(f"Best iteration: {metrics['best_iteration']}")
        
        log.info(f"Training completed. Train Accuracy: {metrics['train_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted class labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.model.get_params()
    
    def set_params(self, **params) -> None:
        """Set model parameters"""
        self.model.set_params(**params)
