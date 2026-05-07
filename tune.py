"""
Hyperparameter tuning with Optuna.
Searches for the best XGBoost params, saves them to models/best_params.json,
then retrains and saves the final model.

Usage:
    python tune.py                        # 50 trials, default symbols
    python tune.py --trials 100           # more trials
    python tune.py --metric val_precision # optimize precision instead of AUC
    python tune.py --symbols AAPL MSFT   # specific symbols
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import optuna
import optuna.logging

from src.utils.logger import log
from src.utils.config import settings
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.data.features import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.models.trainer import ModelTrainer


def load_data(symbols: list, lookback_days: int):
    from datetime import datetime, timedelta
    start_date = datetime.now() - timedelta(days=lookback_days)
    fetcher = DataFetcher(use_alpaca=False)
    df = fetcher.fetch(symbols=symbols, start_date=start_date)

    validator = DataValidator()
    _, _ = validator.validate(df)
    df = validator.clean_data(df)

    fe = FeatureEngineer(window=settings.feature_window)
    df = fe.engineer_features(df, create_target=True, target_horizon=settings.prediction_horizon)

    log.info(f"Data ready: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def make_objective(df, metric: str):
    trainer = ModelTrainer(mlflow_tracking=False)
    train_df, val_df, _ = trainer.time_series_split(df, train_size=0.7, val_size=0.15)
    X_train, y_train = trainer.prepare_features_target(train_df)
    X_val, y_val = trainer.prepare_features_target(val_df)

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # High n_estimators — early stopping decides the actual count
            "n_estimators": 1000,
        }

        model = XGBoostModel(params=params)
        model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=30, verbose=False)

        metrics = trainer.calculate_metrics(
            y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1], prefix="val_"
        )
        return metrics[metric]

    return objective


def main(symbols, lookback_days, n_trials, metric):
    log.info("=" * 70)
    log.info("HYPERPARAMETER TUNING")
    log.info(f"Trials: {n_trials}  |  Metric: {metric} (maximize)")
    log.info("=" * 70)

    df = load_data(symbols, lookback_days)

    # Suppress per-trial Optuna output — we print our own summary
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(df, metric), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    log.info("\nBest trial:")
    log.info(f"  {metric}: {best.value:.4f}")
    log.info("  Params:")
    for k, v in best.params.items():
        log.info(f"    {k}: {v}")

    # Save best params so train.py can load them later
    params_path = Path("models/best_params.json")
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(best.params, f, indent=2)
    log.info(f"\nBest params saved → {params_path}")

    # Retrain with best params on full train+val, evaluate on test
    log.info("\nRetraining final model with best params...")
    final_model = XGBoostModel(model_name="xgboost_tuned", params=best.params)
    trainer = ModelTrainer(mlflow_tracking=True)

    all_metrics = trainer.train_model(
        model=final_model,
        df=df,
        target_col="target_direction",
        train_size=0.7,
        val_size=0.15,
        log_mlflow=True,
        early_stopping_rounds=30,
        verbose=False,
    )

    # Save
    model_dir = Path("models/saved")
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model.save(str(model_dir / f"model_tuned_{ts}.pkl"))
    final_model.save(str(model_dir / "latest_model.pkl"))
    log.info(f"Saved tuned model → models/saved/latest_model.pkl")

    log.info("\n" + "=" * 70)
    log.info("FINAL MODEL PERFORMANCE")
    log.info(f"{'Metric':<20} {'Train':>10} {'Val':>10} {'Test':>10}")
    log.info("-" * 55)
    for m in ["accuracy", "precision", "auc"]:
        log.info(
            f"{m.capitalize():<20}"
            f" {all_metrics.get(f'train_{m}', float('nan')):>10.4f}"
            f" {all_metrics.get(f'val_{m}', float('nan')):>10.4f}"
            f" {all_metrics.get(f'test_{m}', float('nan')):>10.4f}"
        )
    log.info("=" * 70)
    log.info("View all metrics in MLflow: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--metric",
        default="val_auc",
        choices=["val_auc", "val_accuracy", "val_precision"],
        help="Validation metric to maximize",
    )
    args = parser.parse_args()

    main(
        symbols=args.symbols or settings.symbol_list,
        lookback_days=args.lookback_days or settings.lookback_days,
        n_trials=args.trials,
        metric=args.metric,
    )