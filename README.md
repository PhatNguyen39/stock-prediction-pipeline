# Stock Prediction Pipeline

Production-ready ML pipeline for stock price direction prediction using XGBoost, with architecture designed for easy ensemble extension.

## 🎯 Features

- **Automated Data Pipeline**: Fetches stock data from Yahoo Finance or Alpaca
- **Feature Engineering**: 40+ technical indicators and price features
- **Time-Series Validation**: Proper walk-forward validation to prevent data leakage
- **XGBoost Model**: Fast, accurate gradient boosting classifier
- **Ensemble-Ready**: Architecture supports adding LightGBM and CatBoost
- **Web Dashboard**: Built-in browser UI for predictions and model management
- **FastAPI Serving**: Production REST API with async support
- **Fly.io Deployment**: One-command cloud deploy with auto-stop to minimize cost
- **MLflow Tracking**: Experiment tracking and model versioning
- **Docker Support**: Containerized deployment
- **Comprehensive Monitoring**: Logging, metrics, and health checks

## 📁 Project Structure

```
stock-prediction-pipeline/
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Data fetching (Yahoo Finance, Alpaca)
│   │   ├── features.py         # Feature engineering
│   │   └── validator.py        # Data validation & cleaning
│   ├── models/
│   │   ├── base.py             # Abstract model interface
│   │   ├── xgboost_model.py    # XGBoost implementation
│   │   ├── ensemble.py         # Ensemble framework (ready for extension)
│   │   └── trainer.py          # Training with proper time-series validation
│   ├── api/
│   │   ├── main.py             # FastAPI application
│   │   ├── templates/
│   │   │   └── dashboard.html  # Web dashboard UI
│   │   └── static/
│   │       ├── css/style.css   # Dashboard styling
│   │       └── js/app.js       # Dashboard client logic
│   └── utils/
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging setup
├── data/                       # Data storage
├── models/                     # Saved models
├── logs/                       # Application logs
├── mlruns/                     # MLflow tracking
├── train.py                    # Training script
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── fly.toml                    # Fly.io deployment config
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd stock-prediction-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your preferences
# - Add Alpaca API keys (optional, Yahoo Finance works without keys)
# - Configure symbols, training intervals, etc.
```

### 3. Train Your First Model

```bash
# Train with default settings (AAPL, MSFT, GOOGL, AMZN, TSLA)
python train.py

# Or specify custom symbols
python train.py --symbols AAPL MSFT TSLA --lookback-days 365
```

Expected output:
```
[Step 1/5] Fetching stock data...
[Step 2/5] Validating and cleaning data...
[Step 3/5] Engineering features...
[Step 4/5] Training XGBoost model...
[Step 5/5] Saving model...

TRAINING COMPLETED SUCCESSFULLY
Test Accuracy:  0.5423
Test AUC:       0.5891
```

### 4. Start API Server

```bash
# Development mode
uvicorn src.api.main:app --reload

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 5. Open the Web Dashboard

The app includes a built-in dashboard — no separate frontend setup needed.

```bash
# Start the server, then open in your browser:
http://localhost:8000/dashboard
```

From the dashboard you can:
- **See model status** — green/red indicator shows whether a model is loaded
- **Train the model** — click "Train Model" and watch the spinner poll until complete
- **Get predictions** — enter comma-separated symbols (e.g. `AAPL, MSFT, GOOGL`), hit submit, and see results as styled cards with direction arrows, probability %, and confidence badges

All the JSON API endpoints (`/predict`, `/train`, `/health`, etc.) continue to work alongside the dashboard.

### 6. Make Predictions

```bash
# Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"]}'
```

Response:
```json
{
  "predictions": [
    {
      "symbol": "AAPL",
      "current_price": 185.23,
      "predicted_direction": 1,
      "probability": 0.6234,
      "confidence": "medium",
      "timestamp": "2025-02-10T10:30:00"
    }
  ],
  "model_name": "xgboost",
  "model_last_trained": "2025-02-10T09:00:00"
}
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## ☁️ Deploy to Fly.io

Host the app in the cloud so it stays alive when your computer is off. Fly.io auto-stops the machine when idle, so cost is ~$1-5/month.

### Prerequisites

Install the Fly CLI: https://fly.io/docs/flyctl/install/

```bash
fly auth login
```

### First-time setup

```bash
# Create the app
fly apps create stock-prediction-gb

# Create a persistent volume for saved models (1 GB)
fly volumes create models_data --region sjc --size 1

# Set API keys if you use Alpaca (optional — Yahoo Finance works without keys)
fly secrets set ALPACA_API_KEY=your_key ALPACA_SECRET_KEY=your_secret
```

### Deploy

```bash
# Deploy (or use: make deploy)
fly deploy
```

Once deployed, open `https://stock-prediction-gb.fly.dev/dashboard`.

On first deploy there is no pre-trained model in the image — click **Train Model** in the dashboard to train one. The trained model is saved to the persistent volume and will survive future deploys.

### Useful Fly commands

```bash
fly status              # Check app status
fly logs                # Stream logs
fly ssh console         # SSH into the running machine
fly scale memory 2048   # Increase memory if training is slow
```

## 📊 API Endpoints

### Health Check
```bash
GET /health
GET /
```

### Web Dashboard
```bash
GET /dashboard
```

### Get Predictions
```bash
POST /predict
Body: {"symbols": ["AAPL", "MSFT"]}
```

### Train Model
```bash
POST /train
```

### Training Status (poll during training)
```bash
GET /training/status
```

### Model Info
```bash
GET /model/info
```

## 🎓 Understanding the Pipeline

### Data Flow

1. **Data Fetching**: Pulls OHLCV data from Yahoo Finance/Alpaca
2. **Validation**: Checks for missing values, duplicates, anomalies
3. **Feature Engineering**:
   - Price features (returns, momentum, spreads)
   - Volume features (changes, ratios)
   - Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic, ATR)
   - Lag features (historical values)
   - Time features (day of week, month, cyclical encoding)
4. **Time-Series Split**: 70% train, 15% validation, 15% test (chronological)
5. **Training**: XGBoost with early stopping
6. **Evaluation**: Comprehensive metrics (accuracy, AUC, precision, recall, F1)

### Key Features

**Prevents Data Leakage**:
- Time-series aware split (no future data in training)
- Features use only past information
- Proper walk-forward validation

**Production Ready**:
- Comprehensive logging
- Error handling
- Model versioning
- Health checks
- Monitoring metrics

## 🔧 Customization

### Change Stock Symbols

Edit `.env`:
```bash
SYMBOLS=AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META
```

### Adjust Training Parameters

In `train.py` or when calling the API:
```python
# More training data
LOOKBACK_DAYS=730  # 2 years

# Different prediction horizon
PREDICTION_HORIZON=5  # Predict 5 days ahead

# Different feature window
FEATURE_WINDOW=30  # 30-day indicators
```

### Add Custom Features

Edit `src/data/features.py`:
```python
def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add your custom features here"""
    df['my_feature'] = ...  # Your logic
    return df
```

## 🎯 Adding Ensemble (Future Extension)

The architecture is ready for ensemble models:

```python
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel  # To be created
from src.models.catboost_model import CatBoostModel  # To be created
from src.models.ensemble import EnsembleModel

# Create individual models
xgb = XGBoostModel()
lgbm = LightGBMModel()
cat = CatBoostModel()

# Create ensemble
ensemble = EnsembleModel(
    models=[xgb, lgbm, cat],
    weights=[0.4, 0.3, 0.3]  # Or None for equal weights
)

# Train ensemble (trains all models)
trainer.train_model(ensemble, df_features)
```

To add LightGBM:
1. Create `src/models/lightgbm_model.py` (similar structure to `xgboost_model.py`)
2. Implement the `BaseModel` interface
3. Use in ensemble as shown above

## 📈 MLflow Tracking

View experiment tracking:
```bash
mlflow ui --backend-store-uri ./mlruns
```

Open http://localhost:5000 to see:
- Training metrics over time
- Model parameters
- Feature importance
- Model comparisons

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## 📝 Performance Expectations

**Baseline Performance** (5 tech stocks, 1-year data):
- Accuracy: 52-58% (better than random 50%)
- AUC: 0.55-0.65
- Training time: 1-3 minutes
- Prediction latency: <100ms

**Note**: Stock prediction is inherently difficult. Focus on:
1. **No data leakage**: Proper validation prevents overfitting
2. **Risk management**: Use probabilities for position sizing
3. **Ensemble benefit**: Typically +2-5% accuracy improvement
4. **Production quality**: Clean code, monitoring, error handling


## 🔍 Common Issues

**Issue**: Model accuracy around 50%
- **Solution**: Stock prediction is hard! Focus on proper validation and risk management

**Issue**: Training fails with memory error
- **Solution**: Reduce `LOOKBACK_DAYS` or `SYMBOLS` in `.env`

**Issue**: API returns 503
- **Solution**: Train a model first using `python train.py` or `POST /train`

**Issue**: Yahoo Finance data fetch fails
- **Solution**: Some symbols might be delisted, check symbol validity

## 📚 Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Time Series Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

## 🤝 Contributing

This is a portfolio project. Feel free to:
- Add more features
- Implement ensemble models
- Add more data sources
- Improve documentation

## 📄 License

MIT License - free to use for portfolio and learning purposes.

---

**Built for ML Engineer Portfolio**
Demonstrates production-grade ML pipeline development with focus on proper validation, clean architecture, and extensibility.
