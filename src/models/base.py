"""
Base model interface for single models and ensembles
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for prediction models"""
    
    def __init__(self, model_name: str):
        """
        Initialize base model
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability predictions array
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model to disk
        
        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, save_path)
    
    def load(self, path: str) -> None:
        """
        Load model from disk
        
        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            return importance_df.sort_values('importance', ascending=False)
        
        return None
