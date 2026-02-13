"""
Ensemble model combining multiple base models
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import accuracy_score, roc_auc_score
from src.models.base import BaseModel
from src.utils.logger import log


class EnsembleModel(BaseModel):
    """Ensemble model using weighted averaging"""
    
    def __init__(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None,
        model_name: str = "ensemble"
    ):
        """
        Initialize ensemble model
        
        Args:
            models: List of base models to ensemble
            weights: Weights for each model (equal weights if None)
            model_name: Model identifier
        """
        super().__init__(model_name)
        
        self.models = models
        
        # Set equal weights if not provided
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        log.info(f"Ensemble initialized with {len(models)} models")
        log.info(f"Model weights: {dict(zip([m.model_name for m in models], self.weights))}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train all models in the ensemble
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            **kwargs: Additional parameters for base models
            
        Returns:
            Dictionary with ensemble training metrics
        """
        log.info(f"Training ensemble with {len(self.models)} models")
        
        self.feature_names = list(X_train.columns)
        all_metrics = {}
        
        # Train each model
        for i, model in enumerate(self.models):
            log.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            
            model_metrics = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **kwargs
            )
            
            all_metrics[model.model_name] = model_metrics
        
        self.is_trained = True
        
        # Calculate ensemble metrics
        ensemble_metrics = {}
        
        # Training metrics
        y_train_pred = self.predict(X_train)
        y_train_proba = self.predict_proba(X_train)[:, 1]
        
        ensemble_metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        ensemble_metrics['train_auc'] = roc_auc_score(y_train, y_train_proba)
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            y_val_proba = self.predict_proba(X_val)[:, 1]
            
            ensemble_metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
            ensemble_metrics['val_auc'] = roc_auc_score(y_val, y_val_proba)
            
            log.info(f"Ensemble Validation Accuracy: {ensemble_metrics['val_accuracy']:.4f}")
            log.info(f"Ensemble Validation AUC: {ensemble_metrics['val_auc']:.4f}")
        
        # Store individual model metrics
        ensemble_metrics['individual_models'] = all_metrics
        
        return ensemble_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using weighted ensemble
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        # Get probabilities from each model
        probas = self.predict_proba(X)
        
        # Return class with highest probability
        return (probas[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using weighted averaging
        
        Args:
            X: Features for prediction
            
        Returns:
            Weighted average probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        # Collect predictions from all models
        all_probas = []
        for model in self.models:
            probas = model.predict_proba(X)
            all_probas.append(probas)
        
        # Weighted average
        weighted_proba = np.zeros_like(all_probas[0])
        for proba, weight in zip(all_probas, self.weights):
            weighted_proba += weight * proba
        
        return weighted_proba
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get aggregated feature importance from all models
        
        Returns:
            DataFrame with weighted feature importance
        """
        if not self.is_trained:
            return None
        
        all_importances = []
        
        for model, weight in zip(self.models, self.weights):
            importance = model.get_feature_importance()
            if importance is not None:
                importance['weighted_importance'] = importance['importance'] * weight
                importance['model'] = model.model_name
                all_importances.append(importance)
        
        if not all_importances:
            return None
        
        # Aggregate importances
        combined = pd.concat(all_importances)
        aggregated = combined.groupby('feature')['weighted_importance'].sum().reset_index()
        aggregated.columns = ['feature', 'importance']
        
        return aggregated.sort_values('importance', ascending=False)
    
    def save(self, path: str) -> None:
        """Save ensemble and all base models"""
        from pathlib import Path
        import joblib
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save each base model
        for i, model in enumerate(self.models):
            model_path = save_path.parent / f"{save_path.stem}_{model.model_name}{save_path.suffix}"
            model.save(str(model_path))
        
        # Save ensemble metadata
        ensemble_data = {
            'model_name': self.model_name,
            'weights': self.weights,
            'model_names': [m.model_name for m in self.models],
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, save_path)
    
    def load(self, path: str) -> None:
        """Load ensemble and all base models"""
        from pathlib import Path
        import joblib
        
        save_path = Path(path)
        
        # Load ensemble metadata
        ensemble_data = joblib.load(save_path)
        
        self.model_name = ensemble_data['model_name']
        self.weights = ensemble_data['weights']
        self.feature_names = ensemble_data['feature_names']
        self.is_trained = ensemble_data['is_trained']
        
        # Load each base model
        # Note: This requires models to be initialized first
        # In practice, you'd need to reinstantiate models before loading
        log.warning("Ensemble.load() requires models to be pre-initialized")
