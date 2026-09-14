"""Application configuration for AerQualitas."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable application-level configuration."""

    name: str = "AerQualitas"
    version: str = "0.8.0"
    description: str = (
        "Evaluación de la calidad del aire mediante estimación de PM2.5 "
        "con una red neuronal feedforward."
    )
    dataset_filename: str = "PRSA_data_2010.1.1-2014.12.31.csv"
    processed_dataset_filename: str = "beijing_pm25_clean.csv"
    quality_report_filename: str = "data_quality_report.json"
    split_manifest_filename: str = "split_manifest.json"
    baseline_metrics_filename: str = "baseline_metrics.json"
    validation_predictions_filename: str = "baseline_validation_predictions.csv"
    baseline_model_filename: str = "baseline_linear.joblib"
    preprocessor_filename: str = "preprocessor.joblib"
    input_metadata_filename: str = "input_metadata.json"

    mlp_model_filename: str = "aerqualitas_mlp.keras"
    mlp_metrics_filename: str = "mlp_metrics.json"
    mlp_history_filename: str = "mlp_training_history.csv"
    mlp_validation_predictions_filename: str = "mlp_validation_predictions.csv"
    mlp_metadata_filename: str = "mlp_metadata.json"
    mlp_summary_filename: str = "mlp_model_summary.txt"

    final_test_metrics_filename: str = "final_test_metrics.json"
    final_test_predictions_filename: str = "final_test_predictions.csv"
    final_error_analysis_filename: str = "final_error_analysis.json"
    final_error_by_quartile_filename: str = "final_error_by_quartile.csv"
    final_model_manifest_filename: str = "final_model_manifest.json"
    inference_metadata_filename: str = "inference_metadata.json"

    random_seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    mlp_hidden_1: int = 64
    mlp_hidden_2: int = 32
    mlp_learning_rate: float = 0.001
    mlp_batch_size: int = 64
    mlp_max_epochs: int = 200
    mlp_early_stopping_patience: int = 15
    mlp_reduce_lr_patience: int = 6


SETTINGS = AppConfig()
