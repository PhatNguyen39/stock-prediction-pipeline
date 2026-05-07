"""
Walk-forward cross-validation script.
Gives a realistic estimate of out-of-sample performance across multiple time periods
instead of relying on a single train/val/test split.

Usage:
    python validate.py                          # 5 folds, best_params.json if it exists
    python validate.py --folds 8                # more folds
    python validate.py --params-file models/best_params.json
    python validate.py --no-params              # use XGBoost defaults
"""
import argparse
import json
from pathlib import Path

from src.utils.logger import log
from src.utils.config import settings
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.data.features import FeatureEngineer
from src.models.trainer import ModelTrainer


def load_data(symbols: list):
    fetcher = DataFetcher(use_alpaca=False)
    df = fetcher.fetch(symbols=symbols)

    validator = DataValidator()
    _, _ = validator.validate(df)
    df = validator.clean_data(df)

    fe = FeatureEngineer(window=settings.feature_window)
    df = fe.engineer_features(df, create_target=True, target_horizon=settings.prediction_horizon)

    log.info(f"Data ready: {df.shape[0]} rows, prediction_horizon={settings.prediction_horizon}d")
    return df


def main(symbols, n_splits, params_file, no_params):
    log.info("=" * 70)
    log.info("WALK-FORWARD CROSS-VALIDATION")
    log.info(f"Folds: {n_splits}  |  Symbols: {symbols}")
    log.info("=" * 70)

    df = load_data(symbols)

    # Load params
    params = None
    if not no_params:
        path = Path(params_file)
        if path.exists():
            with open(path) as f:
                params = json.load(f)
            log.info(f"Using params from {path}")
        else:
            log.info(f"{path} not found — using XGBoost defaults")

    trainer = ModelTrainer(mlflow_tracking=False)
    fold_metrics, avg_metrics = trainer.walk_forward_validate(
        df=df,
        model_params=params,
        n_splits=n_splits,
        gap=settings.prediction_horizon,
    )

    log.info("\nPer-fold detail:")
    log.info(f"{'Fold':<6} {'Train period':<30} {'Test period':<30}")
    log.info("-" * 68)
    for m in fold_metrics:
        log.info(f"{m['fold']:<6} {m['train_period']:<30} {m['test_period']:<30}")

    log.info("\nSummary:")
    log.info(f"  Avg Accuracy:  {avg_metrics['avg_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    log.info(f"  Avg Precision: {avg_metrics['avg_precision']:.4f} ± {avg_metrics['std_precision']:.4f}")
    log.info(f"  Avg AUC:       {avg_metrics['avg_auc']:.4f} ± {avg_metrics['std_auc']:.4f}")
    log.info("=" * 70)

    return fold_metrics, avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--params-file", type=str, default="models/best_params.json")
    parser.add_argument("--no-params", action="store_true",
                        help="Ignore params file, use XGBoost defaults")
    args = parser.parse_args()

    main(
        symbols=args.symbols or settings.symbol_list,
        n_splits=args.folds,
        params_file=args.params_file,
        no_params=args.no_params,
    )