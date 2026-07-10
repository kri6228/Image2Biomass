"""
Metadata feature engineering utilities for the CSIRO Image2Biomass project.

This module provides reusable functionality for:

- Validating image-level metadata columns.
- Building scikit-learn metadata preprocessing pipelines.
- Processing categorical and numerical metadata variables.
- Generating fold-safe out-of-fold metadata features.
- Preventing cross-validation leakage during preprocessing.
- Returning consistent metadata feature names.

The metadata preprocessing pipeline is fitted independently for each
cross-validation fold using only the corresponding training samples.
Validation samples are transformed using the fitted training-fold
preprocessor.
"""

from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# METADATA COLUMN CONFIGURATION
# ============================================================


CATEGORICAL_METADATA_COLUMNS = [
    "State",
    "Species",
]


NUMERICAL_METADATA_COLUMNS = [
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
]


METADATA_COLUMNS = (
    CATEGORICAL_METADATA_COLUMNS
    + NUMERICAL_METADATA_COLUMNS
)


# ============================================================
# METADATA VALIDATION
# ============================================================


def validate_metadata_columns(
    metadata: pd.DataFrame,
    required_columns: Sequence[str] = METADATA_COLUMNS,
) -> dict:
    """
    Validate metadata columns before feature engineering.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Image-level metadata table.

    required_columns : Sequence[str], default=METADATA_COLUMNS
        Metadata columns required for feature engineering.

    Returns
    -------
    dict
        Metadata validation statistics.

    Raises
    ------
    TypeError
        If metadata is not a pandas DataFrame.

    ValueError
        If required columns are missing or the metadata table
        contains zero rows.
    """

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(
            "metadata must be a pandas DataFrame."
        )

    if len(metadata) == 0:
        raise ValueError(
            "metadata must contain at least one row."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required metadata columns: {missing_columns}"
        )

    missing_value_counts = (
        metadata[list(required_columns)]
        .isna()
        .sum()
        .to_dict()
    )

    validation_results = {
        "n_samples": len(metadata),
        "n_metadata_columns": len(required_columns),
        "missing_value_counts": missing_value_counts,
        "total_missing_values": int(
            sum(missing_value_counts.values())
        ),
    }

    return validation_results


# ============================================================
# METADATA PREPROCESSOR
# ============================================================
def get_categorical_vocabularies(
    metadata: pd.DataFrame,
) -> dict[str, list]:
    """
    Extract the complete categorical vocabularies from metadata.

    These vocabularies define a consistent output feature schema across
    cross-validation folds. Only category identities are shared; fitted
    preprocessing statistics are still learned independently from each
    training fold.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Image-level metadata table.

    Returns
    -------
    dict[str, list]
        Mapping from categorical column names to sorted category values.
    """

    vocabularies = {}

    for column in CATEGORICAL_METADATA_COLUMNS:
        vocabularies[column] = sorted(
            metadata[column]
            .dropna()
            .unique()
            .tolist()
        )

    return vocabularies

def build_metadata_preprocessor(
    categorical_vocabularies: dict[str, list] | None = None,
) -> ColumnTransformer:
    """
    Build the metadata preprocessing transformer.

    Numerical features are median imputed and standardized.

    Categorical features are most-frequent imputed and one-hot encoded
    using a consistent category schema when vocabularies are provided.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    if categorical_vocabularies is None:
        encoder_categories = "auto"
    else:
        encoder_categories = [
            categorical_vocabularies[column]
            for column in CATEGORICAL_METADATA_COLUMNS
        ]

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    categories=encoder_categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_METADATA_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_METADATA_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# FEATURE NAME EXTRACTION
# ============================================================


def get_metadata_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """
    Retrieve output feature names from a fitted metadata preprocessor.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Fitted metadata preprocessing transformer.

    Returns
    -------
    list[str]
        Ordered metadata feature names.
    """

    if not hasattr(preprocessor, "transformers_"):
        raise ValueError(
            "The metadata preprocessor must be fitted before "
            "feature names can be retrieved."
        )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    return feature_names


# ============================================================
# FOLD-SAFE OOF FEATURE GENERATION
# ============================================================


def generate_oof_metadata_features(
    metadata: pd.DataFrame,
    fold_column: str = "fold",
) -> tuple[np.ndarray, list[str], dict]:
    """
    Generate fold-safe out-of-fold metadata features.

    For every cross-validation fold:

    1. Training rows are selected using all other folds.
    2. A new metadata preprocessor is created.
    3. The preprocessor is fitted only on training rows.
    4. Validation rows are transformed using the fitted preprocessor.
    5. Validation features are written back to their original row
       positions.

    This prevents validation-fold information from leaking into fitted
    preprocessing transformations.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Image-level metadata table containing metadata variables and
        fold assignments.

    fold_column : str, default="fold"
        Name of the cross-validation fold column.

    Returns
    -------
    tuple
        oof_features : numpy.ndarray
            Fold-safe metadata feature matrix aligned with the original
            metadata row order.

        feature_names : list[str]
            Ordered output metadata feature names.

        fitted_preprocessors : dict
            Dictionary mapping fold numbers to their fitted metadata
            preprocessors.

    Raises
    ------
    ValueError
        If the fold column is missing, metadata validation fails,
        feature dimensions differ across folds, or some rows do not
        receive out-of-fold features.
    """

    validate_metadata_columns(metadata)

    if fold_column not in metadata.columns:
        raise ValueError(
            f"Fold column '{fold_column}' is missing."
        )

    folds = sorted(
        metadata[fold_column]
        .unique()
        .tolist()
    )

    if len(folds) < 2:
        raise ValueError(
            "At least two folds are required for "
            "out-of-fold feature generation."
        )

    oof_features = None
    reference_feature_names = None

    fitted_preprocessors = {}

    assigned_rows = np.zeros(
        len(metadata),
        dtype=bool,
    )
    
    categorical_vocabularies = get_categorical_vocabularies(
        metadata
    )

    for fold in folds:

        train_mask = (
            metadata[fold_column] != fold
        )

        validation_mask = (
            metadata[fold_column] == fold
        )

        train_metadata = metadata.loc[
            train_mask,
            METADATA_COLUMNS,
        ]

        validation_metadata = metadata.loc[
            validation_mask,
            METADATA_COLUMNS,
        ]

        preprocessor = build_metadata_preprocessor(
            categorical_vocabularies=categorical_vocabularies
        )

        preprocessor.fit(train_metadata)

        validation_features = preprocessor.transform(
            validation_metadata
        )

        validation_features = np.asarray(
            validation_features,
            dtype=np.float32,
        )

        fold_feature_names = get_metadata_feature_names(
            preprocessor
        )

        if reference_feature_names is None:

            reference_feature_names = fold_feature_names

            oof_features = np.empty(
                (
                    len(metadata),
                    len(reference_feature_names),
                ),
                dtype=np.float32,
            )

        elif fold_feature_names != reference_feature_names:

            raise ValueError(
                "Metadata feature names or dimensions differ "
                f"for fold {fold}. This may occur when a "
                "categorical level is absent from a training fold."
            )

        validation_indices = np.flatnonzero(
            validation_mask.to_numpy()
        )

        oof_features[
            validation_indices
        ] = validation_features

        assigned_rows[
            validation_indices
        ] = True

        fitted_preprocessors[fold] = preprocessor

    if not assigned_rows.all():
        unassigned_count = int(
            (~assigned_rows).sum()
        )

        raise ValueError(
            f"{unassigned_count} metadata rows did not receive "
            "out-of-fold features."
        )

    if not np.isfinite(oof_features).all():
        raise ValueError(
            "Generated metadata features contain "
            "NaN or infinite values."
        )

    return (
        oof_features,
        reference_feature_names,
        fitted_preprocessors,
    )