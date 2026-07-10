"""
visual_features.py
===================

Reusable, deterministic handcrafted visual feature extraction for the
Image2Biomass project (Notebook 05).

This module extracts scientifically-motivated, interpretable visual
features from top-view pasture RGB images:

    - RGB / HSV color statistics
    - Excess Green Index (ExG) based vegetation proxy
    - Vegetation-only color statistics
    - Brown / dead-material proxy mask (NOT ground-truth segmentation)
    - Compact color histograms
    - GLCM texture features
    - Canny-edge-based structural/canopy-complexity proxy

IMPORTANT SCOPE NOTES
----------------------
- This module does NOT implement semantic segmentation (U-Net /
  DeepLabV3+). The "vegetation mask" and "brown proxy mask" produced
  here are simple, explicit, rule-based heuristics intended as a
  baseline visual representation, not learned pixel-level ground truth.
- This module does NOT use target/label information. All functions
  operate purely on image pixel data.
- This module does NOT train any model.

All functions are deterministic given the same input image and
configuration parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


# ============================================================
# Validation / loading
# ============================================================

def validate_image_path(image_path: Union[str, Path]) -> Path:
    """
    Validate that an image path exists and is a file.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.

    Returns
    -------
    Path
        The validated path, resolved to a Path object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a file.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {path}")
    return path


def load_visual_image(
    image_path: Union[str, Path],
    target_size: Sequence[int] = (512, 512),
) -> np.ndarray:
    """
    Load an image from disk, convert to RGB, and resize deterministically.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    target_size : Sequence[int]
        (width, height) to resize the image to.

    Returns
    -------
    np.ndarray
        RGB image array of shape (height, width, 3), dtype uint8.

    Raises
    ------
    FileNotFoundError
        If the image path is invalid.
    ValueError
        If the image cannot be decoded by OpenCV.
    """
    path = validate_image_path(image_path)

    bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Failed to decode image at path: {path}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    width, height = target_size
    resized = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)

    return resized


# ============================================================
# 1. RGB color statistics
# ============================================================

def compute_rgb_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    Compute per-channel mean/std/median statistics for an RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys rgb_r_mean, rgb_r_std, rgb_r_median,
        rgb_g_mean, rgb_g_std, rgb_g_median,
        rgb_b_mean, rgb_b_std, rgb_b_median.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")

    image_float = image.astype(np.float64)
    channel_names = ("r", "g", "b")

    stats: Dict[str, float] = {}
    for idx, name in enumerate(channel_names):
        channel = image_float[:, :, idx]
        stats[f"rgb_{name}_mean"] = float(np.mean(channel))
        stats[f"rgb_{name}_std"] = float(np.std(channel))
        stats[f"rgb_{name}_median"] = float(np.median(channel))

    return stats


# ============================================================
# 2. HSV color statistics
# ============================================================

def compute_hsv_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    Compute per-channel mean/std/median statistics for an image in HSV space.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys hsv_h_mean, hsv_h_std, hsv_h_median,
        hsv_s_mean, hsv_s_std, hsv_s_median,
        hsv_v_mean, hsv_v_std, hsv_v_median.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)
    channel_names = ("h", "s", "v")

    stats: Dict[str, float] = {}
    for idx, name in enumerate(channel_names):
        channel = hsv_image[:, :, idx]
        stats[f"hsv_{name}_mean"] = float(np.mean(channel))
        stats[f"hsv_{name}_std"] = float(np.std(channel))
        stats[f"hsv_{name}_median"] = float(np.median(channel))

    return stats


# ============================================================
# 3. Excess Green Index (ExG)
# ============================================================

def compute_excess_green(image: np.ndarray) -> np.ndarray:
    """
    Compute the Excess Green Index (ExG) map for an RGB image.

    ExG is computed on normalized chromaticity coordinates:
        r = R / (R + G + B), g = G / (R + G + B), b = B / (R + G + B)
        ExG = 2g - r - b

    This normalization makes ExG more robust to illumination changes
    than computing 2G - R - B directly on raw pixel intensities.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        Float64 ExG map of shape (H, W), values approximately in [-1, 1].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")

    image_float = image.astype(np.float64)
    r_channel = image_float[:, :, 0]
    g_channel = image_float[:, :, 1]
    b_channel = image_float[:, :, 2]

    total = r_channel + g_channel + b_channel
    # Avoid division by zero for pure-black pixels.
    safe_total = np.where(total == 0, 1.0, total)

    r_norm = r_channel / safe_total
    g_norm = g_channel / safe_total
    b_norm = b_channel / safe_total

    exg = (2.0 * g_norm) - r_norm - b_norm
    # Force pure-black pixels (total == 0) to a neutral ExG of 0.
    exg = np.where(total == 0, 0.0, exg)

    return exg


def summarize_excess_green(exg: np.ndarray) -> Dict[str, float]:
    """
    Summarize an ExG map into compact scalar statistics.

    Parameters
    ----------
    exg : np.ndarray
        ExG map produced by compute_excess_green.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys exg_mean, exg_std, exg_median, exg_min, exg_max.
    """
    return {
        "exg_mean": float(np.mean(exg)),
        "exg_std": float(np.std(exg)),
        "exg_median": float(np.median(exg)),
        "exg_min": float(np.min(exg)),
        "exg_max": float(np.max(exg)),
    }


# ============================================================
# 4. Vegetation mask (ExG-based proxy, NOT semantic segmentation)
# ============================================================

def create_vegetation_mask(exg: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Create a binary vegetation proxy mask by thresholding the ExG map.

    NOTE: This is a simple rule-based heuristic proxy for "green
    vegetation pixels", not a learned or ground-truth semantic
    segmentation mask.

    Parameters
    ----------
    exg : np.ndarray
        ExG map produced by compute_excess_green.
    threshold : float
        Pixels with ExG > threshold are classified as vegetation.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W), True where vegetation is detected.
    """
    return exg > threshold


def compute_vegetation_coverage_ratio(vegetation_mask: np.ndarray) -> float:
    """
    Compute the fraction of pixels classified as vegetation.

    Parameters
    ----------
    vegetation_mask : np.ndarray
        Boolean mask produced by create_vegetation_mask.

    Returns
    -------
    float
        Vegetation coverage ratio in [0, 1]. Returns 0.0 for an
        empty mask (should not occur given non-empty images, but
        handled safely).
    """
    total_pixels = vegetation_mask.size
    if total_pixels == 0:
        return 0.0
    return float(np.sum(vegetation_mask)) / float(total_pixels)


# ============================================================
# 5. Vegetation-only color statistics
# ============================================================

def compute_vegetation_features(
    image: np.ndarray,
    vegetation_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute vegetation coverage ratio and compact color statistics
    restricted to pixels inside the vegetation mask.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    vegetation_mask : np.ndarray
        Boolean mask of shape (H, W) produced by create_vegetation_mask.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        vegetation_coverage_ratio,
        vegetation_rgb_r_mean, vegetation_rgb_g_mean, vegetation_rgb_b_mean,
        vegetation_h_mean, vegetation_s_mean, vegetation_v_mean.

        If the mask contains zero pixels, all vegetation-only color
        statistics are set to 0.0 (coverage ratio will also be 0.0),
        never raising an exception or producing NaN.
    """
    coverage_ratio = compute_vegetation_coverage_ratio(vegetation_mask)

    features: Dict[str, float] = {"vegetation_coverage_ratio": coverage_ratio}

    pixel_count = int(np.sum(vegetation_mask))

    if pixel_count == 0:
        features.update(
            {
                "vegetation_rgb_r_mean": 0.0,
                "vegetation_rgb_g_mean": 0.0,
                "vegetation_rgb_b_mean": 0.0,
                "vegetation_h_mean": 0.0,
                "vegetation_s_mean": 0.0,
                "vegetation_v_mean": 0.0,
            }
        )
        return features

    image_float = image.astype(np.float64)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)

    features["vegetation_rgb_r_mean"] = float(np.mean(image_float[:, :, 0][vegetation_mask]))
    features["vegetation_rgb_g_mean"] = float(np.mean(image_float[:, :, 1][vegetation_mask]))
    features["vegetation_rgb_b_mean"] = float(np.mean(image_float[:, :, 2][vegetation_mask]))

    features["vegetation_h_mean"] = float(np.mean(hsv_image[:, :, 0][vegetation_mask]))
    features["vegetation_s_mean"] = float(np.mean(hsv_image[:, :, 1][vegetation_mask]))
    features["vegetation_v_mean"] = float(np.mean(hsv_image[:, :, 2][vegetation_mask]))

    return features


# ============================================================
# 6. Brown / dead-material proxy mask
# ============================================================

def create_brown_proxy_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a simple HSV-based proxy mask for brown / dead vegetation
    material.

    IMPORTANT: This is an explicit, rule-based heuristic proxy only.
    It is NOT ground-truth semantic segmentation and should not be
    interpreted as a validated dead-material classifier. It is
    intended purely as a compact, interpretable baseline visual
    signal.

    The heuristic flags pixels with hue in a typical brown/tan/straw
    range and moderate-to-low saturation/value, which tends to
    correspond to dried/senescent plant material and bare soil in
    top-view pasture imagery.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W), True where the brown/dead-material
        proxy criterion is met.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv_image[:, :, 0].astype(np.float64)        # OpenCV hue range: [0, 179]
    saturation = hsv_image[:, :, 1].astype(np.float64)  # [0, 255]
    value = hsv_image[:, :, 2].astype(np.float64)       # [0, 255]

    # Brown/tan/straw hues roughly fall between yellow and orange-red
    # in OpenCV's [0, 179] hue scale (approx 5-35 degrees of a 0-360 scale).
    hue_mask = (hue >= 5) & (hue <= 35)
    # Dead material tends to be less saturated and moderately bright
    # compared to vivid green vegetation.
    saturation_mask = (saturation >= 30) & (saturation <= 180)
    value_mask = value >= 60

    brown_mask = hue_mask & saturation_mask & value_mask
    return brown_mask


def compute_brown_proxy_features(
    image: np.ndarray,
    brown_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute brown/dead-material proxy coverage ratio and compact
    color statistics restricted to the brown proxy mask.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    brown_mask : np.ndarray
        Boolean mask produced by create_brown_proxy_mask.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        brown_coverage_ratio,
        brown_rgb_r_mean, brown_rgb_g_mean, brown_rgb_b_mean.

        If the mask contains zero pixels, color statistics are set
        to 0.0 safely.
    """
    total_pixels = brown_mask.size
    coverage_ratio = float(np.sum(brown_mask)) / float(total_pixels) if total_pixels > 0 else 0.0

    features: Dict[str, float] = {"brown_coverage_ratio": coverage_ratio}

    pixel_count = int(np.sum(brown_mask))
    if pixel_count == 0:
        features.update(
            {
                "brown_rgb_r_mean": 0.0,
                "brown_rgb_g_mean": 0.0,
                "brown_rgb_b_mean": 0.0,
            }
        )
        return features

    image_float = image.astype(np.float64)
    features["brown_rgb_r_mean"] = float(np.mean(image_float[:, :, 0][brown_mask]))
    features["brown_rgb_g_mean"] = float(np.mean(image_float[:, :, 1][brown_mask]))
    features["brown_rgb_b_mean"] = float(np.mean(image_float[:, :, 2][brown_mask]))

    return features


# ============================================================
# 7. Color histogram features
# ============================================================

def compute_color_histograms(
    image: np.ndarray,
    bins: int = 16,
) -> Dict[str, float]:
    """
    Compute compact, normalized HSV color histogram features.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    bins : int
        Number of histogram bins per channel.

    Returns
    -------
    Dict[str, float]
        Dictionary with deterministic keys:
        hist_h_00 ... hist_h_{bins-1},
        hist_s_00 ... hist_s_{bins-1},
        hist_v_00 ... hist_v_{bins-1}.
        Each channel's histogram is L1-normalized to sum to 1.0.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    channel_ranges = {
        "h": (0, 180),
        "s": (0, 256),
        "v": (0, 256),
    }

    features: Dict[str, float] = {}
    for channel_idx, (channel_name, (low, high)) in enumerate(channel_ranges.items()):
        channel_data = hsv_image[:, :, channel_idx]
        hist, _ = np.histogram(channel_data, bins=bins, range=(low, high))
        hist = hist.astype(np.float64)
        hist_sum = hist.sum()
        normalized_hist = hist / hist_sum if hist_sum > 0 else hist

        for bin_idx, value in enumerate(normalized_hist):
            features[f"hist_{channel_name}_{bin_idx:02d}"] = float(value)

    return features


# ============================================================
# 8. GLCM texture features
# ============================================================

def compute_texture_features(
    image: np.ndarray,
    distances: Sequence[int] = (1, 3, 5),
    angles: Sequence[float] = (0.0,),
) -> Dict[str, float]:
    """
    Compute compact GLCM (Gray-Level Co-occurrence Matrix) texture
    statistics, aggregated across the given distances and angles.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    distances : Sequence[int]
        Pixel pair distances for GLCM computation.
    angles : Sequence[float]
        Pixel pair angles (radians) for GLCM computation.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        texture_contrast, texture_dissimilarity, texture_homogeneity,
        texture_energy, texture_correlation.
        Each value is the mean of that property across all
        (distance, angle) combinations, keeping the feature set compact
        regardless of how many distances/angles are configured.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Reduce gray levels to keep the GLCM computation efficient and stable.
    n_gray_levels = 32
    gray_reduced = (gray.astype(np.float64) / 255.0 * (n_gray_levels - 1)).astype(np.uint8)

    glcm = graycomatrix(
        gray_reduced,
        distances=list(distances),
        angles=list(angles),
        levels=n_gray_levels,
        symmetric=True,
        normed=True,
    )

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    features: Dict[str, float] = {}
    for prop in properties:
        prop_values = graycoprops(glcm, prop)  # shape (n_distances, n_angles)
        features[f"texture_{prop}"] = float(np.mean(prop_values))

    return features


# ============================================================
# 9. Edge / canopy-complexity proxy
# ============================================================

def compute_edge_features(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> Dict[str, float]:
    """
    Compute a Canny-edge-density feature as a structural /
    canopy-complexity proxy.

    NOTE: This is a 2D structural-complexity proxy derived from edge
    density in the RGB image, NOT a measurement of actual 3D canopy
    density or height.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    low_threshold : int
        Lower threshold for the Canny edge detector.
    high_threshold : int
        Upper threshold for the Canny edge detector.

    Returns
    -------
    Dict[str, float]
        Dictionary with key: edge_density (fraction of edge pixels).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    total_pixels = edges.size
    edge_density = float(np.sum(edges > 0)) / float(total_pixels) if total_pixels > 0 else 0.0

    return {"edge_density": edge_density}


# ============================================================
# Full per-image feature extraction
# ============================================================

def extract_visual_features_from_image(
    image_path: Union[str, Path],
    target_size: Sequence[int] = (512, 512),
    exg_threshold: float = 0.0,
    hist_bins: int = 16,
    glcm_distances: Sequence[int] = (1, 3, 5),
    glcm_angles: Sequence[float] = (0.0,),
) -> Dict[str, Union[str, float]]:
    """
    Extract the complete set of handcrafted visual features for a
    single image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    target_size : Sequence[int]
        (width, height) to resize the image to before extraction.
    exg_threshold : float
        Threshold for the ExG-based vegetation mask.
    hist_bins : int
        Number of histogram bins per HSV channel.
    glcm_distances : Sequence[int]
        Pixel distances for GLCM texture computation.
    glcm_angles : Sequence[float]
        Pixel angles (radians) for GLCM texture computation.

    Returns
    -------
    Dict[str, Union[str, float]]
        Dictionary of all extracted features, plus an "image_path" key
        holding the original (string) path for downstream alignment.
        All numeric values are guaranteed finite (no NaN or inf).

    Raises
    ------
    FileNotFoundError
        If the image path is invalid.
    ValueError
        If the image cannot be decoded.
    """
    path = validate_image_path(image_path)
    image = load_visual_image(path, target_size=target_size)

    features: Dict[str, Union[str, float]] = {"image_path": str(image_path)}

    features.update(compute_rgb_statistics(image))
    features.update(compute_hsv_statistics(image))

    exg = compute_excess_green(image)
    features.update(summarize_excess_green(exg))

    vegetation_mask = create_vegetation_mask(exg, threshold=exg_threshold)
    features.update(compute_vegetation_features(image, vegetation_mask))

    brown_mask = create_brown_proxy_mask(image)
    features.update(compute_brown_proxy_features(image, brown_mask))

    features.update(compute_color_histograms(image, bins=hist_bins))
    features.update(
        compute_texture_features(image, distances=glcm_distances, angles=glcm_angles)
    )
    features.update(compute_edge_features(image))

    # Final finite-value safety check for this image's features.
    for key, value in features.items():
        if key == "image_path":
            continue
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite value produced for feature '{key}' on image '{image_path}': {value}"
            )

    return features


# ============================================================
# Batch extraction
# ============================================================

def extract_visual_features_batch(
    image_paths: Sequence[Union[str, Path]],
    target_size: Sequence[int] = (512, 512),
    exg_threshold: float = 0.0,
    hist_bins: int = 16,
    glcm_distances: Sequence[int] = (1, 3, 5),
    glcm_angles: Sequence[float] = (0.0,),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract handcrafted visual features for a batch of images,
    preserving the original input order.

    Parameters
    ----------
    image_paths : Sequence[str or Path]
        Ordered sequence of image paths to process.
    target_size : Sequence[int]
        (width, height) to resize each image to before extraction.
    exg_threshold : float
        Threshold for the ExG-based vegetation mask.
    hist_bins : int
        Number of histogram bins per HSV channel.
    glcm_distances : Sequence[int]
        Pixel distances for GLCM texture computation.
    glcm_angles : Sequence[float]
        Pixel angles (radians) for GLCM texture computation.
    verbose : bool
        If True, print simple progress information every 50 images.

    Returns
    -------
    pd.DataFrame
        One row per image, in the same order as `image_paths`, with an
        "image_path" column for alignment and all extracted feature
        columns.
    """
    records: List[Dict[str, Union[str, float]]] = []

    for idx, image_path in enumerate(image_paths):
        record = extract_visual_features_from_image(
            image_path,
            target_size=target_size,
            exg_threshold=exg_threshold,
            hist_bins=hist_bins,
            glcm_distances=glcm_distances,
            glcm_angles=glcm_angles,
        )
        records.append(record)

        if verbose and (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images...")

    if verbose:
        print(f"Finished processing {len(image_paths)}/{len(image_paths)} images.")

    feature_table = pd.DataFrame.from_records(records)

    # Preserve exact input order explicitly (defensive, in case any
    # future change to this function introduces reordering).
    feature_table["__original_order__"] = range(len(image_paths))
    feature_table = feature_table.sort_values("__original_order__").drop(
        columns="__original_order__"
    )
    feature_table = feature_table.reset_index(drop=True)

    # Move image_path to the first column for readability.
    columns = ["image_path"] + [c for c in feature_table.columns if c != "image_path"]
    feature_table = feature_table[columns]

    return feature_table


# ============================================================
# Validation
# ============================================================

def validate_visual_feature_table(
    feature_table: pd.DataFrame,
    expected_row_count: int = None,
) -> None:
    """
    Validate a visual feature table for basic integrity.

    Checks:
        - "image_path" column is present.
        - No missing (NaN) values.
        - No infinite values in numeric columns.
        - No duplicate image_path values.
        - Row count matches expected_row_count, if provided.

    Parameters
    ----------
    feature_table : pd.DataFrame
        The visual feature table to validate.
    expected_row_count : int, optional
        If provided, the exact number of rows the table must have.

    Raises
    ------
    ValueError
        If any validation check fails. The error message specifies
        which check failed.
    """
    if "image_path" not in feature_table.columns:
        raise ValueError("Visual feature table is missing required 'image_path' column.")

    if feature_table.isna().any().any():
        offending_columns = feature_table.columns[feature_table.isna().any()].tolist()
        raise ValueError(
            f"Visual feature table contains missing values in columns: {offending_columns}"
        )

    numeric_columns = feature_table.select_dtypes(include=[np.number]).columns
    numeric_values = feature_table[numeric_columns].to_numpy()
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("Visual feature table contains non-finite (inf/-inf) values.")

    duplicate_count = feature_table["image_path"].duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Visual feature table contains {duplicate_count} duplicate image_path values."
        )

    if expected_row_count is not None and len(feature_table) != expected_row_count:
        raise ValueError(
            f"Visual feature table has {len(feature_table)} rows; "
            f"expected {expected_row_count}."
        )