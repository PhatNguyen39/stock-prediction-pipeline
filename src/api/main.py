"""
FastAPI application for stock prediction serving
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from pathlib import Path

TRAINING_DISABLED = os.environ.get("DISABLE_TRAINING", "").lower() == "true"

from src.utils.config import settings
from src.utils.logger import log
from src.data.fetcher import DataFetcher
from src.data.features import FeatureEngineer
from src.data.validator import DataValidator
from src.models.xgboost_model import XGBoostModel
from src.models.trainer import ModelTrainer

# Paths for templates and static files
_BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

@asynccontextmanager
async def lifespan(app):
    """Initialize model on startup"""
    log.info("Starting Stock Prediction API")
    log.info(f"Configuration: {settings.model_type}")

    # Initialize components
    state.data_fetcher = DataFetcher(use_alpaca=False)
    state.feature_engineer = FeatureEngineer(window=settings.feature_window)

    # Try to load existing model
    model_path = Path("models/saved/latest_model.pkl")
    if model_path.exists():
        try:
            log.info("Loading existing model")
            state.model = XGBoostModel()
            state.model.load(str(model_path))
            state.is_ready = True
            log.info("Model loaded successfully")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            state.model = None
    else:
        log.warning("No trained model found. Use /train endpoint to train a model.")

    yield


# Initialize FastAPI
app = FastAPI(
    title="Stock Prediction API",
    description="ML-powered stock price direction prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")

# Global state
class ModelState:
    def __init__(self):
        self.model: Optional[XGBoostModel] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.data_fetcher: Optional[DataFetcher] = None
        self.last_trained: Optional[datetime] = None
        self.is_ready: bool = False
        self.training_in_progress: bool = False
        self.training_error: Optional[str] = None

state = ModelState()


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    symbols: List[str] = Field(..., description="Stock symbols to predict")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{"symbols": ["AAPL", "MSFT", "GOOGL"]}]
        }
    }


class PredictionResult(BaseModel):
    """Single prediction result"""
    symbol: str
    current_price: float
    predicted_direction: int  # 0 = down, 1 = up
    probability: float
    confidence: str  # "high", "medium", "low"
    timestamp: datetime


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_config = {"protected_namespaces": ()}

    predictions: List[PredictionResult]
    model_name: str
    model_last_trained: Optional[datetime]


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_name: Optional[str]
    last_trained: Optional[datetime]
    uptime_seconds: float


class TrainingStatus(BaseModel):
    """Training status response"""
    status: str
    message: str
    metrics: Optional[Dict] = None


# Health check
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    import time
    
    return HealthResponse(
        status="healthy" if state.is_ready else "not_ready",
        model_loaded=state.model is not None and state.is_ready,
        model_name=state.model.model_name if state.model else None,
        last_trained=state.last_trained,
        uptime_seconds=time.time()  # Simplified, in production use actual start time
    )


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "ok"}


@app.get("/dashboard")
async def dashboard(request: Request):
    """Serve the web dashboard"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "training_enabled": not TRAINING_DISABLED,
    })


@app.get("/training/status")
async def training_status():
    """Poll training progress"""
    if TRAINING_DISABLED:
        return {"status": "disabled", "message": "Training is disabled on this deployment. Train locally and push the model."}
    if state.training_in_progress:
        return {"status": "in_progress", "message": "Training is running..."}
    if state.training_error:
        error = state.training_error
        state.training_error = None
        return {"status": "failed", "message": error}
    if state.is_ready:
        return {"status": "complete", "message": "Model is trained and ready."}
    return {"status": "idle", "message": "No model trained yet."}


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions for given stock symbols
    
    Returns predictions for price direction (up/down)
    """
    if not state.is_ready or state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Please train a model first using /train endpoint"
        )
    
    try:
        log.info(f"Prediction request for symbols: {request.symbols}")
        
        # Fetch recent data for prediction
        end_date = datetime.now()
        start_date = end_date - timedelta(days=settings.lookback_days)
        
        df = state.data_fetcher.fetch(
            symbols=request.symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate data
        validator = DataValidator()
        df = validator.clean_data(df)
        
        # Engineer features (without creating target for inference)
        df_features = state.feature_engineer.engineer_features(
            df,
            create_target=False,
            target_horizon=settings.prediction_horizon
        )
        
        # Get the most recent data point for each symbol
        latest_data = df_features.sort_values('date').groupby('symbol').tail(1)
        
        # Prepare features for prediction
        exclude_cols = [
            'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'
        ]
        feature_cols = [col for col in latest_data.columns if col not in exclude_cols]
        X = latest_data[feature_cols]
        
        # Make predictions
        predictions = state.model.predict(X)
        probabilities = state.model.predict_proba(X)[:, 1]
        
        # Prepare response
        results = []
        for idx, row in latest_data.iterrows():
            symbol_idx = latest_data.index.get_loc(idx)
            prob = float(probabilities[symbol_idx])
            pred = int(predictions[symbol_idx])
            
            # Determine confidence level
            if prob > 0.7 or prob < 0.3:
                confidence = "high"
            elif prob > 0.6 or prob < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            results.append(
                PredictionResult(
                    symbol=row['symbol'],
                    current_price=float(row['close']),
                    predicted_direction=pred,
                    probability=prob,
                    confidence=confidence,
                    timestamp=datetime.now()
                )
            )
        
        log.info(f"Predictions generated for {len(results)} symbols")
        
        return PredictionResponse(
            predictions=results,
            model_name=state.model.model_name,
            model_last_trained=state.last_trained
        )
    
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoint
@app.post("/train", response_model=TrainingStatus)
async def train_model(background_tasks: BackgroundTasks):
    """
    Trigger model training

    Training happens in the background and model is updated upon completion
    """
    if TRAINING_DISABLED:
        return TrainingStatus(
            status="disabled",
            message="Training is disabled on this deployment. Train locally and push the model."
        )
    if state.model is not None and state.is_ready:
        # Check if recently trained
        if state.last_trained and (datetime.now() - state.last_trained).total_seconds() < 3600:
            return TrainingStatus(
                status="skipped",
                message="Model was recently trained. Wait at least 1 hour between trainings."
            )
    
    # Add training task to background
    background_tasks.add_task(run_training)
    
    return TrainingStatus(
        status="started",
        message="Model training started in background. Check /health for status."
    )


async def run_training():
    """Background task to train the model"""
    state.training_in_progress = True
    state.training_error = None
    try:
        log.info("Starting model training")
        
        # Fetch data
        fetcher = DataFetcher(use_alpaca=False)
        df = fetcher.fetch()
        
        # Validate and clean
        validator = DataValidator()
        df = validator.clean_data(df)
        
        # Engineer features
        feature_engineer = FeatureEngineer(window=settings.feature_window)
        df_features = feature_engineer.engineer_features(
            df,
            create_target=True,
            target_horizon=settings.prediction_horizon
        )
        
        # Train model
        model = XGBoostModel(model_name="xgboost")
        trainer = ModelTrainer(mlflow_tracking=True)
        
        metrics = trainer.train_model(
            model=model,
            df=df_features,
            target_col='target_direction',
            train_size=0.7,
            val_size=0.15,
            log_mlflow=True
        )
        
        # Save model
        model_path = Path("models/saved")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(f"models/saved/model_{timestamp}.pkl")
        
        # Save as latest
        model.save("models/saved/latest_model.pkl")
        
        # Update state
        state.model = model
        state.feature_engineer = feature_engineer
        state.last_trained = datetime.now()
        state.is_ready = True
        
        log.info("Model training completed successfully")
        log.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")

    except Exception as e:
        log.error(f"Training failed: {e}")
        state.training_error = str(e)
    finally:
        state.training_in_progress = False


# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get information about the current model"""
    if not state.is_ready or state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importance = state.model.get_feature_importance()
    
    return {
        "model_name": state.model.model_name,
        "is_trained": state.model.is_trained,
        "last_trained": state.last_trained,
        "n_features": len(state.model.feature_names) if state.model.feature_names else 0,
        "top_features": importance.head(10).to_dict('records') if importance is not None else []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
