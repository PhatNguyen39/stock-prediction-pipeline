---
title: Tech Stock Prediction with Technical Analysis
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Tech Stock Prediction with Technical Analysis

> **Live Demo:** [Stock Prediction Dashboard](https://PhatNguyen39-Stock-Prediction-Pipeline.hf.space/dashboard)

> **Disclaimer:** This project is for **demonstration and portfolio purposes only**. **This is not financial advice and should not be used for trading decisions.**

Production-ready ML pipeline for tech stock price direction prediction using XGBoost, with a full experiment tracking and hyperparameter tuning workflow.

## Scope

The model is trained on **5 large-cap tech stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA using **5 years** of historical data.

Predictions are **5 trading days ahead** (weekly direction).

Features include both technical indicators and external market signals (VIX, SPY, XLK sector ETF, earnings proximity). Expected test accuracy is 52–58% — this is the honest, expected result for large-cap equities due to the Efficient Market Hypothesis. See [Performance Expectations](#-performance-expectations) for more.

## 🎯 Features

- **Automated Data Pipeline**: Fetches OHLCV data from Yahoo Finance (5 years lookback)
- **Feature Engineering**: 47+ features across technical indicators, external market signals, and time features
- **External Market Signals**: VIX fear index, SPY/XLK returns, days to next earnings
- **Hyperparameter Tuning**: Optuna-based search with 50+ trials, saves best params automatically
- **Walk-Forward Validation**: 5-fold expanding-window cross-validation for honest performance estimates
- **Time-Series Safe**: Chronological train/val/test split, no data leakage
- **XGBoost Classifier**: Binary classification (UP/DOWN), probability output with confidence labels
- **Web Dashboard**: Built-in browser UI — pick from the 5 supported stocks
- **FastAPI Serving**: Production REST API, prediction uses only 90 days of data (fast)
- **MLflow Tracking**: Full experiment history — params, metrics, feature importance per run
- **Docker Support**: Containerized deployment
- **Hugging Face Spaces**: Free cloud deployment with Docker SDK

## 📁 Project Structure

```
stock-prediction-pipeline/
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Data fetching (Yahoo Finance)
│   │   ├── features.py         # Feature engineering + external signals
│   │   └── validator.py        # Data validation & cleaning
│   ├── models/
│   │   ├── base.py             # Abstract model interface
│   │   ├── xgboost_model.py    # XGBoost implementation
│   │   ├── ensemble.py         # Ensemble framework (ready for extension)
│   │   └── trainer.py          # Training + walk-forward validation
│   ├── api/
│   │   ├── main.py             # FastAPI application
│   │   ├── templates/
│   │   │   └── dashboard.html  # Web dashboard UI
│   │   └── static/
│   │       ├── css/style.css
│   │       └── js/app.js
│   └── utils/
│       ├── config.py           # Configuration (lookback, horizon, symbols)
│       └── logger.py           # Logging setup
├── models/
│   ├── saved/
│   │   └── latest_model.pkl    # Active model loaded by the API
│   └── best_params.json        # Best hyperparams from last tune.py run
├── data/                       # Data storage (gitignored)
├── logs/                       # Application logs (gitignored)
├── mlruns/                     # MLflow experiment history (gitignored)
├── train.py                    # Train with best_params.json (auto-loaded)
├── tune.py                     # Optuna hyperparameter search + retrain
├── validate.py                 # Walk-forward cross-validation
├── check_importance.py         # Print feature importance from saved model
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repo-url>
cd stock-prediction-pipeline

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Recommended Workflow

#### Option A — Tune then train (best results)

```bash
# Step 1: Search for best hyperparameters (50 trials, ~10–20 min)
python tune.py

# Step 2: Validate performance across 5 market regimes
python validate.py

# Step 3: Train is not required — tune.py already saves the model.
#         Re-run train.py only if you want to retrain on fresh data
#         using the saved best params:
python train.py
```

#### Option B — Train directly with defaults

```bash
python train.py
```

`train.py` automatically loads `models/best_params.json` if it exists.
If the file is missing or empty, it falls back to XGBoost defaults.

### 3. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Open the Dashboard

```
http://localhost:8000/dashboard
```

From the dashboard you can:
- **See model status** — indicator shows whether a model is loaded
- **Train the model** — click "Train Model" and poll until complete
- **Get predictions** — toggle stocks, hit submit, see direction + probability + confidence badge

Only `medium` and `high` confidence predictions (probability > 60%) are worth acting on.

### 5. Make Predictions via API

```bash
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
      "timestamp": "2026-05-06T10:30:00"
    }
  ],
  "model_name": "xgboost",
  "model_last_trained": "2026-05-06T09:00:00"
}
```

**Confidence labels:**
- `high` → probability > 70%
- `medium` → probability 60–70%
- `low` → probability ≤ 60% — ignore these signals

## 🔧 Scripts

### `tune.py` — Hyperparameter search

```bash
python tune.py                            # 50 trials, optimise val_precision
python tune.py --trials 100               # more trials = better search
python tune.py --metric val_auc           # optimise AUC instead
python tune.py --symbols AAPL MSFT        # specific symbols
```

Runs Optuna search, saves best params to `models/best_params.json`, retrains the final model, saves to `models/saved/latest_model.pkl`.

### `train.py` — Training

```bash
python train.py                           # auto-loads best_params.json
python train.py --symbols AAPL MSFT TSLA  # specific symbols
python train.py --params-file other.json  # explicit params file
```

### `validate.py` — Walk-forward validation

```bash
python validate.py           # 5 folds with best_params.json
python validate.py --folds 8 # more folds
python validate.py --no-params  # XGBoost defaults
```

Reports per-fold and average accuracy/precision/AUC with std across market regimes. Does **not** save a model — purely diagnostic.

### `check_importance.py` — Feature importance

```bash
python check_importance.py
```

Prints all features ranked by XGBoost importance from the current `latest_model.pkl`.

## 🧠 Understanding the Pipeline

### Data Flow

```
Yahoo Finance (5 years OHLCV)
        ↓
Validation & Cleaning
        ↓
Feature Engineering (47+ features)
        ↓
Walk-forward / Train-Val-Test Split
        ↓
XGBoost Binary Classifier
        ↓
REST API → Dashboard
```

### Features (47+)

**Price** — 1d/5d/10d/20d returns, momentum, HL spread, close position in range

**Volume** — volume change, volume/MA ratio, Volume-Price Trend (VPT)

**Technical Indicators** — SMA(20), EMA(20), MACD, RSI(14), Bollinger Bands (width + position), Stochastic K/D, ATR(14)

**Lag Features** — close, volume, return lagged at [1, 2, 3, 5, 10] days

**Time** — day of week (+ cyclical encoding), day of month, month, quarter

**External Market Signals** (fetched live during training and prediction):
- `vix` — CBOE VIX fear index (market-wide risk regime)
- `spy_return_1d`, `spy_return_5d` — broad market momentum
- `xlk_return_1d`, `xlk_return_5d` — tech sector momentum
- `days_to_earnings` — days until next earnings announcement (0–90, capped)

### Prediction at Inference Time

Training uses 5 years of data to learn patterns. Predicting uses only the **last 90 days** to compute feature values for the most recent row — fast and efficient.

### Preventing Data Leakage

- Chronological split only — no shuffling
- A gap equal to the prediction horizon (5 days) between train end and test start in walk-forward folds
- External features fetched using only past data at each point in time

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/dashboard` | Web UI |
| POST | `/predict` | Get predictions for symbols |
| POST | `/train` | Trigger model training |
| GET | `/training/status` | Poll training progress |
| GET | `/model/info` | Current model metadata |

## 📈 MLflow Tracking

Every `train.py` and `tune.py` run is automatically logged.

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

> **macOS note:** Port 5000 is used by AirPlay Receiver. Use `--port 5001`.

Each run stores:
- **Params**: model type, feature count, data sizes
- **Metrics**: accuracy, precision, AUC for train / val / test splits
- **Artifacts**: `feature_importance.csv`, trained model pickle

## 📉 Performance Expectations

| Metric | Typical range |
|---|---|
| Test Accuracy | 52–57% |
| Test Precision | 52–58% |
| Test AUC | 0.51–0.55 |
| Training time | 3–10 min (with tuning) |
| Prediction latency | < 200ms |

**Why not higher?** Large-cap tech stocks (AAPL, MSFT, GOOGL, AMZN, TSLA) are the most heavily analysed equities in the world. Any signal from standard technical indicators is arbitraged away almost instantly by institutional quant funds. An honest 52–57% test precision with no overfitting is the correct, expected result — not a failure.

A model claiming 80%+ accuracy on this problem almost certainly has data leakage.

**What would improve it:**
- Alternative data: news sentiment, options flow, earnings surprises
- Less-efficient markets: mid/small-cap stocks
- Intraday data with microstructure features

## 🔧 Configuration

Key settings in `src/utils/config.py` (can be overridden via `.env`):

```bash
SYMBOLS=AAPL,MSFT,GOOGL,AMZN,TSLA   # stocks to train on
LOOKBACK_DAYS=1825                    # 5 years of training data
PREDICTION_HORIZON=5                  # predict 5 trading days ahead
FEATURE_WINDOW=20                     # rolling window for indicators
```

## 🐳 Docker Deployment

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## ☁️ Deploy to Hugging Face Spaces (Free)

### 1. Train locally and commit the model

```bash
python train.py
git add models/saved/latest_model.pkl models/best_params.json
git commit -m "Update trained model"
git push
```

### 2. Create a Space and sync from GitHub

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space), choose **Docker** SDK
2. Add your HF token as a GitHub secret (`HF_TOKEN`) in **repo Settings → Secrets → Actions**
3. Edit `.github/workflows/sync-to-hf.yml` — replace `YOUR_HF_USERNAME` with your username
4. In the Space **Settings → Variables**, add `DISABLE_TRAINING=true`

The pre-trained model is loaded on startup. To update, retrain locally, commit, and push.

## 🔍 Common Issues

**`403` on `http://localhost:5000` (MLflow UI)**
→ macOS AirPlay uses port 5000. Run `mlflow ui --port 5001` instead.

**All predictions show `low` confidence**
→ Expected when market has no clear directional signal. Only act on `medium`/`high` confidence predictions.

**Accuracy stuck at ~50% / Accuracy equals Precision**
→ Model may be predicting all UP. Re-tune with `python tune.py --metric val_precision`.

**Training fails with memory error**
→ Reduce `LOOKBACK_DAYS` in `.env` or use fewer symbols.

**API returns 503**
→ No model loaded. Run `python train.py` first or call `POST /train`.

## 📚 Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

## 📄 License

MIT License — free to use for portfolio and learning purposes.

---

**Built for ML Engineer Portfolio**
Demonstrates a production-grade ML pipeline: data ingestion, feature engineering with external signals, hyperparameter tuning, honest time-series evaluation, experiment tracking, and REST API serving.