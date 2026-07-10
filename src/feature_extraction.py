"""
Feature extraction utilities for the CSIRO Image2Biomass project.

This module provides reusable functionality for:

- Building a frozen pretrained EfficientNetB0 feature extractor.
- Loading and preprocessing pasture images.
- Creating efficient TensorFlow tf.data input pipelines.
- Extracting fixed-length CNN embeddings.
- Validating generated CNN features.

The extracted EfficientNetB0 embeddings are intended to be cached and
reused by downstream machine learning and deep learning experiments.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0

from src.config import (
    CNN_BATCH_SIZE,
    CNN_FEATURE_DIM,
    IMAGE_SIZE,
)


# ============================================================
# FEATURE EXTRACTOR
# ============================================================


def build_efficientnet_feature_extractor() -> Model:
    """
    Build a frozen EfficientNetB0 feature extractor.

    The model uses ImageNet pretrained weights and excludes the original
    classification head. Global average pooling converts the final
    convolutional feature maps into a fixed-length 1280-dimensional
    embedding.

    Returns
    -------
    tensorflow.keras.Model
        Frozen EfficientNetB0 feature extraction model.
    """

    model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*IMAGE_SIZE, 3),
    )

    model.trainable = False

    return model


# ============================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================


def load_and_preprocess_image(
    image_path: tf.Tensor,
) -> tf.Tensor:
    """
    Load and preprocess a single image.

    Parameters
    ----------
    image_path : tensorflow.Tensor
        Path to the image file.

    Returns
    -------
    tensorflow.Tensor
        Decoded and resized image tensor.
    """

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image


# ============================================================
# TENSORFLOW DATA PIPELINE
# ============================================================


def build_image_dataset(
    image_paths: Sequence[str | Path],
    batch_size: int = CNN_BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Build an efficient TensorFlow dataset for feature extraction.

    The dataset preserves the input image order so that extracted
    embeddings remain aligned with the original metadata table.

    Parameters
    ----------
    image_paths : Sequence[str | pathlib.Path]
        Ordered collection of image paths.

    batch_size : int, default=CNN_BATCH_SIZE
        Number of images processed per batch.

    Returns
    -------
    tensorflow.data.Dataset
        Batched and prefetched TensorFlow dataset.
    """

    if len(image_paths) == 0:
        raise ValueError(
            "image_paths must contain at least one image."
        )

    image_paths = [
        str(Path(path))
        for path in image_paths
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        image_paths
    )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# CNN FEATURE EXTRACTION
# ============================================================


def extract_cnn_features(
    model: Model,
    dataset: tf.data.Dataset,
    verbose: int = 1,
) -> np.ndarray:
    """
    Extract CNN embeddings from an image dataset.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Frozen CNN feature extraction model.

    dataset : tensorflow.data.Dataset
        Batched image dataset.

    verbose : int, default=1
        Keras prediction verbosity level.

    Returns
    -------
    numpy.ndarray
        Extracted CNN embeddings with shape
        (number_of_images, CNN_FEATURE_DIM).
    """

    features = model.predict(
        dataset,
        verbose=verbose,
    )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    return features


# ============================================================
# FEATURE VALIDATION
# ============================================================


def validate_cnn_features(
    features: np.ndarray,
    expected_samples: int,
    expected_dim: int = CNN_FEATURE_DIM,
) -> dict:
    """
    Validate extracted CNN feature embeddings.

    Parameters
    ----------
    features : numpy.ndarray
        CNN feature matrix.

    expected_samples : int
        Expected number of image embeddings.

    expected_dim : int, default=CNN_FEATURE_DIM
        Expected embedding dimension.

    Returns
    -------
    dict
        Dictionary containing feature validation statistics.

    Raises
    ------
    ValueError
        If the feature matrix fails any structural or numerical
        validation checks.
    """

    if not isinstance(features, np.ndarray):
        raise TypeError(
            "features must be a NumPy array."
        )

    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, "
            f"but received shape {features.shape}."
        )

    expected_shape = (
        expected_samples,
        expected_dim,
    )

    if features.shape != expected_shape:
        raise ValueError(
            f"Expected feature shape {expected_shape}, "
            f"but received {features.shape}."
        )

    nan_count = int(
        np.isnan(features).sum()
    )

    inf_count = int(
        np.isinf(features).sum()
    )

    if nan_count > 0:
        raise ValueError(
            f"CNN features contain {nan_count} NaN values."
        )

    if inf_count > 0:
        raise ValueError(
            f"CNN features contain {inf_count} infinite values."
        )

    feature_variances = np.var(
        features,
        axis=0,
    )

    zero_variance_features = int(
        np.sum(feature_variances == 0)
    )

    validation_results = {
        "n_samples": features.shape[0],
        "n_features": features.shape[1],
        "dtype": str(features.dtype),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_variance_features": zero_variance_features,
        "feature_min": float(np.min(features)),
        "feature_max": float(np.max(features)),
        "feature_mean": float(np.mean(features)),
        "feature_std": float(np.std(features)),
    }

    return validation_results