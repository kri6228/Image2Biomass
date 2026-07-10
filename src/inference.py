"""
Production inference utilities for the Image2Biomass project.

This module extracts the validated classical inference pipeline used in
Notebook 08.

Final production strategy:
    EfficientNetB0 CNN features
    + fold-specific metadata features
    + handcrafted visual features
    -> ExtraTrees fold models
    -> average fold predictions
    -> clip negative predictions

The module supports both batch inference and single-sample inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)

from src.modeling import TARGET_COLUMNS

from src.metadata_features import METADATA_COLUMNS

from src.visual_features import (
    extract_visual_features_batch,
)

from src.feature_extraction import (
    extract_cnn_features,
    build_image_dataset,
    build_efficientnet_feature_extractor,
)

from src.advanced_modeling import (
    identify_advanced_feature_columns,
)


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# Paths
# ============================================================

# ============================================================
# Production Artifact Paths
# ============================================================

PRODUCTION_ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "production_artifacts"
)

PRODUCTION_MODEL_DIR = (
    PRODUCTION_ARTIFACTS_DIR
    / "models"
)

METADATA_PREPROCESSOR_DIR = (
    PRODUCTION_ARTIFACTS_DIR
    / "metadata_preprocessors"
)

PRODUCTION_CONFIGURATION_DIR = (
    PRODUCTION_ARTIFACTS_DIR
    / "configuration"
)

CANONICAL_FEATURE_COLUMNS_PATH = (
    PRODUCTION_CONFIGURATION_DIR
    / "canonical_feature_columns.json"
)

BEST_CONFIGURATION_PATH = (
    PRODUCTION_CONFIGURATION_DIR
    / "06_best_configuration.json"
)

FINAL_STRATEGY_PATH = (
    PRODUCTION_CONFIGURATION_DIR
    / "08_final_strategy.json"
)


# ============================================================
# Configuration Loading
# ============================================================

def load_final_strategy(
    strategy_path: Path = FINAL_STRATEGY_PATH,
) -> dict[str, Any]:
    """
    Load the final strategy selected by Notebook 08.
    """

    strategy_path = Path(strategy_path)

    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Final strategy file not found: {strategy_path}"
        )

    with strategy_path.open("r", encoding="utf-8") as file:
        strategy = json.load(file)

    required_keys = {
        "weight_classical",
        "weight_cnn",
    }

    missing_keys = required_keys - set(strategy)

    if missing_keys:
        raise ValueError(
            "Final strategy file is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return strategy


def load_best_configuration(
    configuration_path: Path = BEST_CONFIGURATION_PATH,
) -> dict[str, Any]:
    """
    Load the best Notebook 06 model configuration.
    """

    configuration_path = Path(configuration_path)

    if not configuration_path.exists():
        raise FileNotFoundError(
            "Best configuration file not found: "
            f"{configuration_path}"
        )

    with configuration_path.open("r", encoding="utf-8") as file:
        configuration = json.load(file)

    required_keys = {
        "model",
        "feature_set",
    }

    missing_keys = required_keys - set(configuration)

    if missing_keys:
        raise ValueError(
            "Best configuration is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return configuration


# ============================================================
# Canonical Feature Columns
# ============================================================

def load_canonical_feature_columns(
    feature_set_name: str,
    feature_columns_path: Path = CANONICAL_FEATURE_COLUMNS_PATH,
) -> list[str]:
    """
    Load the exact canonical predictive feature order used
    during model training.

    The production inference pipeline stores this feature order
    as a lightweight JSON artifact so the complete training
    feature table is not required during deployment.
    """

    feature_columns_path = Path(
        feature_columns_path
    )

    if not feature_columns_path.exists():

        raise FileNotFoundError(
            "Canonical feature columns artifact "
            f"not found: {feature_columns_path}"
        )

    with open(
        feature_columns_path,
        "r",
        encoding="utf-8",
    ) as file:

        canonical_columns = json.load(file)

    if not isinstance(
        canonical_columns,
        list,
    ):

        raise TypeError(
            "Canonical feature columns artifact "
            "must contain a JSON list."
        )

    if not canonical_columns:

        raise ValueError(
            "Canonical feature columns artifact "
            "is empty."
        )

    if len(
        canonical_columns
    ) != len(
        set(canonical_columns)
    ):

        raise ValueError(
            "Canonical feature columns artifact "
            "contains duplicate feature names."
        )

    logger.info(
        "Loaded %d canonical feature columns.",
        len(canonical_columns),
    )

    return canonical_columns


# ============================================================
# Artifact Loading
# ============================================================

def get_model_path(
    fold: int,
    model_name: str,
    feature_set_name: str,
) -> Path:
    """
    Build the saved Notebook 06 fold-model path.
    """

    return (
        PRODUCTION_MODEL_DIR
        / (
            f"06_best_"
            f"{model_name.lower()}_"
            f"{feature_set_name.lower()}_"
            f"fold_{fold}.joblib"
        )
    )


def get_metadata_preprocessor_path(
    fold: int,
) -> Path:
    """
    Build the fold-specific metadata preprocessor path.
    """

    return (
        METADATA_PREPROCESSOR_DIR
        / f"metadata_preprocessor_fold_{fold}.joblib"
    )


def load_fold_models(
    model_name: str,
    feature_set_name: str,
    n_folds: int = 5,
) -> list[Any]:
    """
    Load and validate all fitted classical fold models.
    """

    models = []

    for fold in range(n_folds):

        model_path = get_model_path(
            fold=fold,
            model_name=model_name,
            feature_set_name=feature_set_name,
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Fold model not found: {model_path}"
            )

        model = joblib.load(model_path)

        if not hasattr(model, "predict"):
            raise TypeError(
                f"Invalid model artifact at {model_path}. "
                f"Loaded object type: {type(model).__name__}"
            )

        models.append(model)

    logger.info(
        "Loaded and validated %d fold models.",
        len(models),
    )

    return models


def load_metadata_preprocessors(
    n_folds: int = 5,
) -> list[Any]:
    """
    Load all saved fold-specific metadata preprocessors.
    """

    preprocessors = []

    for fold in range(n_folds):

        preprocessor_path = (
            get_metadata_preprocessor_path(fold)
        )

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                "Metadata preprocessor not found: "
                f"{preprocessor_path}"
            )

        preprocessor = joblib.load(preprocessor_path)

        if not hasattr(preprocessor, "transform"):
            raise TypeError(
                "Invalid metadata preprocessor artifact at "
                f"{preprocessor_path}."
            )

        preprocessors.append(preprocessor)

    logger.info(
        "Loaded and validated %d metadata preprocessors.",
        len(preprocessors),
    )

    return preprocessors


# ============================================================
# Artifact Validation
# ============================================================

def validate_inference_artifacts(
    n_folds: int = 5,
) -> dict[str, Any]:
    """
    Validate all artifacts required for production inference.
    """

    strategy = load_final_strategy()
    configuration = load_best_configuration()

    feature_set_name = configuration["feature_set"]
    model_name = configuration["model"]

    canonical_columns = load_canonical_feature_columns(
        feature_set_name
    )

    models = load_fold_models(
        model_name=model_name,
        feature_set_name=feature_set_name,
        n_folds=n_folds,
    )

    preprocessors = load_metadata_preprocessors(
        n_folds=n_folds
    )

    validation_result = {
        "strategy": strategy,
        "configuration": configuration,
        "n_canonical_features": len(canonical_columns),
        "n_models": len(models),
        "n_preprocessors": len(preprocessors),
    }

    logger.info(
        "Inference artifacts validated successfully."
    )

    return validation_result


# ============================================================
# Input Validation
# ============================================================

def validate_image_paths(
    image_paths: Sequence[str | Path],
) -> list[str]:
    """
    Validate image paths and return normalized string paths.
    """

    if len(image_paths) == 0:
        raise ValueError(
            "At least one image path is required."
        )

    validated_paths = []

    for image_path in image_paths:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        validated_paths.append(str(path))

    return validated_paths


def prepare_metadata_dataframe(
    metadata: pd.DataFrame,
    expected_samples: int,
) -> pd.DataFrame:
    """
    Validate and order raw metadata for preprocessing.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            "metadata must be a pandas DataFrame."
        )

    if len(metadata) != expected_samples:
        raise ValueError(
            "Image and metadata sample counts do not match. "
            f"Images: {expected_samples}, "
            f"metadata rows: {len(metadata)}."
        )

    missing_columns = [
        column
        for column in METADATA_COLUMNS
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{missing_columns}"
        )

    return (
        metadata[METADATA_COLUMNS]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# Feature Extraction
# ============================================================

def extract_inference_cnn_features(
    image_paths: Sequence[str | Path],
    feature_extractor: Any | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract EfficientNetB0 CNN features.
    """

    validated_paths = validate_image_paths(
        image_paths
    )

    dataset = build_image_dataset(
        validated_paths,
        batch_size=batch_size,
    )

    if feature_extractor is None:
        feature_extractor = (
            build_efficientnet_feature_extractor()
        )

    cnn_features = extract_cnn_features(
        model=feature_extractor,
        dataset=dataset,
    )

    cnn_features = np.asarray(
        cnn_features,
        dtype=np.float32,
    )

    if cnn_features.ndim != 2:
        raise ValueError(
            "CNN features must be a 2D array."
        )

    if cnn_features.shape[0] != len(validated_paths):
        raise ValueError(
            "CNN feature sample count does not match "
            "the number of images."
        )

    if np.isnan(cnn_features).any():
        raise ValueError(
            "CNN features contain NaN values."
        )

    if np.isinf(cnn_features).any():
        raise ValueError(
            "CNN features contain infinite values."
        )

    cnn_columns = [
        f"cnn_{index:04d}"
        for index in range(cnn_features.shape[1])
    ]

    logger.info(
        "Extracted CNN features with shape %s.",
        cnn_features.shape,
    )

    return cnn_features, cnn_columns


def extract_inference_visual_features(
    image_paths: Sequence[str | Path],
) -> pd.DataFrame:
    """
    Extract handcrafted visual features.
    """

    validated_paths = validate_image_paths(
        image_paths
    )

    visual_features = extract_visual_features_batch(
        validated_paths
    )

    if not isinstance(visual_features, pd.DataFrame):
        visual_features = pd.DataFrame(
            visual_features
        )

    visual_features = (
        visual_features
        .copy()
        .reset_index(drop=True)
    )

    if len(visual_features) != len(validated_paths):
        raise ValueError(
            "Visual feature sample count does not match "
            "the number of images."
        )

    numeric_visual_features = (
        visual_features.select_dtypes(
            include=[np.number]
        )
    )

    if numeric_visual_features.isna().any().any():
        raise ValueError(
            "Visual features contain NaN values."
        )

    if np.isinf(
        numeric_visual_features.to_numpy()
    ).any():
        raise ValueError(
            "Visual features contain infinite values."
        )

    logger.info(
        "Extracted visual features with shape %s.",
        visual_features.shape,
    )

    return visual_features


# ============================================================
# Feature Matrix Construction
# ============================================================

def build_fold_feature_matrix(
    cnn_features: np.ndarray,
    cnn_columns: Sequence[str],
    metadata: pd.DataFrame,
    visual_features: pd.DataFrame,
    metadata_preprocessor: Any,
    canonical_columns: Sequence[str],
) -> np.ndarray:
    """
    Construct and validate one fold's aligned feature matrix.
    """

    sample_count = len(metadata)

    transformed_metadata = (
        metadata_preprocessor.transform(metadata)
    )

    transformed_metadata = np.asarray(
        transformed_metadata,
        dtype=np.float32,
    )

    metadata_columns = (
        metadata_preprocessor
        .get_feature_names_out()
        .tolist()
    )

    cnn_dataframe = pd.DataFrame(
        cnn_features,
        columns=cnn_columns,
        index=range(sample_count),
    )

    metadata_dataframe = pd.DataFrame(
        transformed_metadata,
        columns=metadata_columns,
        index=range(sample_count),
    )

    visual_dataframe = (
        visual_features
        .copy()
        .reset_index(drop=True)
    )

    combined_features = pd.concat(
        [
            cnn_dataframe,
            metadata_dataframe,
            visual_dataframe,
        ],
        axis=1,
    )

    missing_columns = [
        column
        for column in canonical_columns
        if column not in combined_features.columns
    ]

    if missing_columns:
        raise ValueError(
            "Combined inference features are missing "
            f"{len(missing_columns)} canonical columns. "
            f"First missing columns: {missing_columns[:10]}"
        )

    aligned_features = combined_features[
        list(canonical_columns)
    ]

    if aligned_features.shape[1] != len(
        canonical_columns
    ):
        raise ValueError(
            "Aligned feature count is incorrect."
        )

    if aligned_features.isna().any().any():
        raise ValueError(
            "Aligned inference features contain NaN values."
        )

    feature_array = aligned_features.to_numpy(
        dtype=np.float32
    )

    if np.isinf(feature_array).any():
        raise ValueError(
            "Aligned inference features contain "
            "infinite values."
        )

    return feature_array


# ============================================================
# Batch Prediction
# ============================================================

def predict_batch(
    image_paths: Sequence[str | Path],
    metadata: pd.DataFrame,
    feature_extractor: Any | None = None,
    batch_size: int = 32,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Predict all biomass targets for multiple samples.
    """

    validated_paths = validate_image_paths(
        image_paths
    )

    metadata_dataframe = prepare_metadata_dataframe(
        metadata=metadata,
        expected_samples=len(validated_paths),
    )

    strategy = load_final_strategy()

    classical_weight = float(
        strategy["weight_classical"]
    )

    cnn_regression_weight = float(
        strategy["weight_cnn"]
    )

    if classical_weight <= 0.0:
        raise ValueError(
            "The current inference module implements the "
            "selected Notebook 08 classical production "
            "strategy, but its configured classical weight "
            "is not positive."
        )

    if cnn_regression_weight != 0.0:
        raise ValueError(
            "The saved final strategy includes Notebook 07 "
            "CNN regression. This inference module currently "
            "implements the validated selected strategy where "
            "weight_cnn is 0.0."
        )

    configuration = load_best_configuration()

    model_name = configuration["model"]
    feature_set_name = configuration["feature_set"]

    canonical_columns = load_canonical_feature_columns(
        feature_set_name
    )

    models = load_fold_models(
        model_name=model_name,
        feature_set_name=feature_set_name,
        n_folds=n_folds,
    )

    preprocessors = load_metadata_preprocessors(
        n_folds=n_folds
    )

    cnn_features, cnn_columns = (
        extract_inference_cnn_features(
            image_paths=validated_paths,
            feature_extractor=feature_extractor,
            batch_size=batch_size,
        )
    )

    visual_features = (
        extract_inference_visual_features(
            validated_paths
        )
    )

    fold_predictions = []

    for fold, (model, preprocessor) in enumerate(
        zip(models, preprocessors)
    ):

        feature_matrix = build_fold_feature_matrix(
            cnn_features=cnn_features,
            cnn_columns=cnn_columns,
            metadata=metadata_dataframe,
            visual_features=visual_features,
            metadata_preprocessor=preprocessor,
            canonical_columns=canonical_columns,
        )

        predictions = model.predict(
            feature_matrix
        )

        predictions = np.asarray(
            predictions,
            dtype=np.float32,
        )

        predictions = np.clip(
            predictions,
            a_min=0.0,
            a_max=None,
        )

        expected_shape = (
            len(validated_paths),
            len(TARGET_COLUMNS),
        )

        if predictions.shape != expected_shape:
            raise ValueError(
                f"Fold {fold} prediction shape is "
                f"{predictions.shape}; expected "
                f"{expected_shape}."
            )

        if np.isnan(predictions).any():
            raise ValueError(
                f"Fold {fold} predictions contain NaN values."
            )

        if np.isinf(predictions).any():
            raise ValueError(
                f"Fold {fold} predictions contain "
                "infinite values."
            )

        fold_predictions.append(predictions)

        logger.info(
            "Completed inference for fold %d.",
            fold,
        )

    final_predictions = np.mean(
        fold_predictions,
        axis=0,
    )

    final_predictions = np.clip(
        final_predictions,
        a_min=0.0,
        a_max=None,
    )

    prediction_dataframe = pd.DataFrame(
        final_predictions,
        columns=TARGET_COLUMNS,
    )

    logger.info(
        "Completed batch inference for %d samples.",
        len(prediction_dataframe),
    )

    return prediction_dataframe


# ============================================================
# Single-Sample Prediction
# ============================================================

def predict_single_sample(
    image_path: str | Path,
    metadata: Mapping[str, Any],
    feature_extractor: Any | None = None,
) -> dict[str, float]:
    """
    Predict all biomass targets for one image and metadata row.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping of metadata "
            "column names to values."
        )

    metadata_dataframe = pd.DataFrame(
        [dict(metadata)]
    )

    predictions = predict_batch(
        image_paths=[image_path],
        metadata=metadata_dataframe,
        feature_extractor=feature_extractor,
    )

    prediction_row = predictions.iloc[0]

    return {
        target: float(prediction_row[target])
        for target in TARGET_COLUMNS
    }


# ============================================================
# Public API
# ============================================================

__all__ = [
    "load_final_strategy",
    "load_best_configuration",
    "load_canonical_feature_columns",
    "load_fold_models",
    "load_metadata_preprocessors",
    "validate_inference_artifacts",
    "extract_inference_cnn_features",
    "extract_inference_visual_features",
    "build_fold_feature_matrix",
    "predict_batch",
    "predict_single_sample",
]