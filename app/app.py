import sys
import tempfile
from datetime import datetime
from pathlib import Path
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference import (
    predict_single_sample,
    validate_inference_artifacts,
)

from src.visual_features import (
    compute_excess_green,
    create_vegetation_mask,
    load_visual_image,
)

from src.modeling import TARGET_COLUMNS
from src.config import VISUAL_IMAGE_SIZE


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image2Biomass",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

TRAIN_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)


VALID_STATES = [
    "NSW",
    "Tas",
    "Vic",
    "WA",
]


NAV_PAGES = [
    "Home / Prediction",
    "Try a Sample",
    "Monitoring Dashboard",
    "Model Information",
    "About the Project",
]


TARGET_DISPLAY_LABELS = {
    "Dry_Clover_g": "Dry Clover",
    "Dry_Dead_g": "Dry Dead Material",
    "Dry_Green_g": "Dry Green Vegetation",
    "Dry_Total_g": "Total Dry Biomass",
    "GDM_g": "Green Dry Matter",
}


# ============================================================
# CUSTOM CSS
# ============================================================

def render_custom_css():

    st.markdown(
        """
        <style>

        /* --------------------------------------------------
           MAIN LAYOUT
        -------------------------------------------------- */

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }


        /* --------------------------------------------------
           HEADINGS
        -------------------------------------------------- */

        h1 {
            color: #173D25 !important;
            font-weight: 750 !important;
            letter-spacing: -0.025em;
        }

        h2 {
            color: #214E31 !important;
            font-weight: 700 !important;
        }

        h3 {
            color: #2D5A3A !important;
            font-weight: 650 !important;
        }


        /* --------------------------------------------------
           SIDEBAR
        -------------------------------------------------- */

        section[data-testid="stSidebar"] {
            border-right: 1px solid #D9E2DA;
        }


        /* --------------------------------------------------
           HERO
        -------------------------------------------------- */

        .hero-box {
            background:
                linear-gradient(
                    135deg,
                    #EDF5EF 0%,
                    #F8FAF7 100%
                );

            border: 1px solid #D7E5DA;
            border-radius: 18px;

            padding: 2rem 2.2rem;
            margin-bottom: 1.5rem;
        }

        .hero-title {
            color: #173D25;
            font-size: 2.5rem;
            line-height: 1.1;
            font-weight: 750;
            margin-bottom: 0.6rem;
        }

        .hero-subtitle {
            color: #52675A;
            font-size: 1.05rem;
            line-height: 1.6;
            max-width: 850px;
            margin-bottom: 1.1rem;
        }


        /* --------------------------------------------------
           BADGES
        -------------------------------------------------- */

        .badge {
            display: inline-block;

            color: #245533;
            background-color: #E1EFE5;

            border: 1px solid #C8DECD;
            border-radius: 999px;

            padding: 5px 12px;

            margin-right: 6px;
            margin-bottom: 5px;

            font-size: 0.78rem;
            font-weight: 650;
        }


        /* --------------------------------------------------
           SECTION HEADER
        -------------------------------------------------- */

        .section-description {
            color: #617167;
            margin-top: -0.5rem;
            margin-bottom: 1.3rem;
        }


        /* --------------------------------------------------
           METRIC CARDS
        -------------------------------------------------- */

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;

            border: 1px solid #DCE5DD;
            border-radius: 14px;

            padding: 16px;

            box-shadow:
                0 2px 8px
                rgba(25, 60, 35, 0.05);
        }


        /* --------------------------------------------------
           PIPELINE
        -------------------------------------------------- */

        .pipeline-container {
            background-color: #FFFFFF;

            border: 1px solid #DCE5DD;
            border-radius: 16px;

            padding: 1.5rem;

            box-shadow:
                0 2px 8px
                rgba(25, 60, 35, 0.05);
        }

        .pipeline-step {
            background-color: #F2F7F3;

            border: 1px solid #D6E4D9;
            border-radius: 10px;

            padding: 12px 16px;
        }

        .pipeline-step-title {
            color: #173D25;

            font-size: 1rem;
            font-weight: 700;

            margin-bottom: 3px;
        }

        .pipeline-step-description {
            color: #637268;
            font-size: 0.84rem;
        }

        .pipeline-arrow {
            text-align: center;

            color: #2F5233;

            font-size: 1.3rem;
            font-weight: 700;

            padding: 4px;
        }


        /* --------------------------------------------------
           SAMPLE CARDS
        -------------------------------------------------- */

        .sample-title {
            color: #173D25;

            font-size: 1.15rem;
            font-weight: 700;

            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }


        /* --------------------------------------------------
           INFORMATION CARDS
        -------------------------------------------------- */

        .info-card {
            background-color: #FFFFFF;

            border: 1px solid #DCE5DD;
            border-radius: 14px;

            padding: 1.3rem;

            height: 100%;

            box-shadow:
                0 2px 8px
                rgba(25, 60, 35, 0.04);
        }

        .info-card-title {
            color: #214E31;

            font-size: 1.05rem;
            font-weight: 700;

            margin-bottom: 0.5rem;
        }

        .info-card-text {
            color: #617167;

            font-size: 0.9rem;
            line-height: 1.55;
        }


        /* --------------------------------------------------
           BUTTONS
        -------------------------------------------------- */

        .stButton > button {
            border-radius: 9px;
            font-weight: 650;
            min-height: 2.8rem;
        }


        /* --------------------------------------------------
           FILE UPLOADER
        -------------------------------------------------- */

        section[data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
        }


        /* --------------------------------------------------
           DATAFRAME
        -------------------------------------------------- */

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }


        /* --------------------------------------------------
           RESPONSIVE
        -------------------------------------------------- */

        @media (max-width: 768px) {

            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-box {
                padding: 1.4rem;
            }

            .hero-title {
                font-size: 2rem;
            }
        }


        /* =====================================================
            METRIC CARD FIX
        ===================================================== */

            div[data-testid="stMetric"] {
                background-color: #FFFFFF !important;
                border: 1px solid #DCE5DD !important;
                border-radius: 14px !important;
                padding: 18px !important;
            }

            div[data-testid="stMetric"] label,
            div[data-testid="stMetricLabel"],
            div[data-testid="stMetricLabel"] p {
                color: #53665A !important;
            }

            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] > div {
                color: #173D25 !important;
            }

            div[data-testid="stMetricValue"] * {
                color: #173D25 !important;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_training_metadata():

    if not TRAIN_CSV_PATH.exists():
        return pd.DataFrame()

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    required_columns = [
        "image_path",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ]

    if not all(
        column in train_df.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    return (
        train_df
        .drop_duplicates("image_path")
        [required_columns]
        .reset_index(drop=True)
    )


TRAINING_METADATA = load_training_metadata()


if not TRAINING_METADATA.empty:

    SPECIES_OPTIONS = sorted(
        TRAINING_METADATA["Species"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    SPECIES_OPTIONS = [
        "Ryegrass_Clover",
    ]


# ============================================================
# SAMPLE CONFIGURATION
# ============================================================

def build_sample_config(number_of_samples=6):

    if TRAINING_METADATA.empty:
        return []

    valid_samples = []

    for _, row in TRAINING_METADATA.iterrows():

        try:
            relative_path = Path(str(row["image_path"]))

            image_path = (
                PROJECT_ROOT
                / "data"
                / "raw"
                / relative_path
            )

            # Never add unavailable images
            if not image_path.exists():
                continue

            # Never add samples with missing required metadata
            required_values = [
                row["State"],
                row["Species"],
                row["Pre_GSHH_NDVI"],
                row["Height_Ave_cm"],
            ]

            if any(pd.isna(value) for value in required_values):
                continue

            sample = {
                "name": (
                    f"{row['State']} · "
                    f"{str(row['Species']).replace('_', ' ')}"
                ),

                "image_path": image_path,

                "state": str(row["State"]),

                "species": str(row["Species"]),

                "ndvi": float(row["Pre_GSHH_NDVI"]),

                "height_cm": float(row["Height_Ave_cm"]),

                "sampling_area_m2": 1.0,
            }

            valid_samples.append(sample)

        except Exception:
            continue

    # Prefer different species for sample variety
    unique_samples = []
    used_species = set()

    for sample in valid_samples:

        if sample["species"] not in used_species:

            unique_samples.append(sample)

            used_species.add(sample["species"])

        if len(unique_samples) >= number_of_samples:
            break

    # If fewer unique species exist, fill remaining slots
    if len(unique_samples) < number_of_samples:

        used_paths = {
            str(sample["image_path"])
            for sample in unique_samples
        }

        for sample in valid_samples:

            if str(sample["image_path"]) in used_paths:
                continue

            unique_samples.append(sample)

            used_paths.add(str(sample["image_path"]))

            if len(unique_samples) >= number_of_samples:
                break

    return unique_samples


SAMPLE_CONFIG = build_sample_config()


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    
    defaults = {

        "current_page": NAV_PAGES[0],

        "navigation_widget": NAV_PAGES[0],

        "pending_navigation": None,

        "sample_selection_error": None,

        "active_image_source": None,

        "selected_sample_name": None,

        "metadata_state": VALID_STATES[0],

        "metadata_species": SPECIES_OPTIONS[0],

        "metadata_ndvi": 0.50,

        "metadata_height": 10.0,

        "sampling_area": 1.0,

        "last_predictions": None,

        "last_density_predictions": None,

        "last_sampling_area": None,

        "last_image_caption": None,

        "last_vegetation_heatmap": None,

        "last_vegetation_overlay": None,

        "last_vegetation_heatmap_error": None,

        "monitoring_history": [],

        "monitoring_clear_confirmation": False,

        "monitoring_last_committed_token": None,
    }


    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


# ============================================================
# RESOURCE VALIDATION
# ============================================================

@st.cache_resource
def validate_resources():

    return validate_inference_artifacts()


# ============================================================
# GENERAL HELPERS
# ============================================================

def clear_results():

    st.session_state[
        "last_predictions"
    ] = None

    st.session_state[
        "last_density_predictions"
    ] = None

    st.session_state[
        "last_sampling_area"
    ] = None

    st.session_state[
        "last_image_caption"
    ] = None

    st.session_state[
        "last_vegetation_heatmap"
    ] = None

    st.session_state[
        "last_vegetation_overlay"
    ] = None

    st.session_state[
        "last_vegetation_heatmap_error"
    ] = None


def get_selected_sample():

    selected_name = (
        st.session_state.get(
            "selected_sample_name"
        )
    )


    for sample in SAMPLE_CONFIG:

        if (
            sample["name"]
            == selected_name
        ):
            return sample


    return None


def select_sample(sample):

    image_path = Path(sample["image_path"])

    if not image_path.exists():
        st.session_state["sample_selection_error"] = (
            f"Sample image does not exist: {image_path}"
        )
        return

    st.session_state["selected_sample_name"] = sample["name"]

    st.session_state["active_image_source"] = "sample"

    st.session_state["metadata_state"] = sample["state"]

    st.session_state["metadata_species"] = sample["species"]

    st.session_state["metadata_ndvi"] = float(sample["ndvi"])

    st.session_state["metadata_height"] = float(sample["height_cm"])

    st.session_state["sampling_area"] = float(
        sample.get("sampling_area_m2", 1.0)
    )

    st.session_state["sample_selection_error"] = None

    clear_results()

    # Navigation is processed safely at the beginning
    # of the next Streamlit execution.
    st.session_state["pending_navigation"] = "Home / Prediction"


def clear_selected_sample():

    st.session_state[
        "selected_sample_name"
    ] = None

    st.session_state[
        "active_image_source"
    ] = None

    clear_results()


def calculate_kg_ha(
    predictions,
    sampling_area_m2,
):

    return {

        target:
            (
                float(
                    predictions[target]
                )
                / sampling_area_m2
            )
            * 10.0

        for target
        in TARGET_COLUMNS
    }


def build_vegetation_biomass_proxy_heatmap(image):

    if image is None:
        raise ValueError("No image data provided for heatmap generation.")

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Heatmap generation expects a NumPy image array."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB image with shape (H, W, 3), got {image.shape}."
        )

    exg = compute_excess_green(image)
    vegetation_mask = create_vegetation_mask(exg, threshold=0.0)

    lower_percentile, upper_percentile = np.percentile(exg, [2, 98])

    if not np.isfinite(lower_percentile) or not np.isfinite(upper_percentile):
        normalized = np.zeros_like(exg, dtype=np.uint8)
    elif upper_percentile <= lower_percentile:
        normalized = np.zeros_like(exg, dtype=np.uint8)
    else:
        scaled = np.clip(
            (exg - lower_percentile) / (upper_percentile - lower_percentile),
            0.0,
            1.0,
        )
        normalized = (scaled * 255.0).astype(np.uint8)

    normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=7.0)

    heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(image, 0.58, heatmap_rgb, 0.42, 0.0)

    return {
        "heatmap": heatmap_rgb,
        "overlay": overlay_rgb,
        "vegetation_coverage_ratio": float(np.mean(vegetation_mask)),
        "exg_min": float(np.min(exg)),
        "exg_max": float(np.max(exg)),
    }


def append_monitoring_record(
    image_source_name,
    image_source_type,
    metadata,
    sampling_area,
    predictions,
    density_predictions,
    prediction_event_token,
):

    timestamp = datetime.now().isoformat(timespec="seconds")

    if (
        st.session_state.get("monitoring_last_committed_token")
        == prediction_event_token
    ):

        return


    history = st.session_state["monitoring_history"]

    history.append(
        {
            "Timestamp": timestamp,
            "Image Source Name": image_source_name,
            "Image Source Type": image_source_type,
            "State": metadata["State"],
            "Species": metadata["Species"],
            "Pre_GSHH_NDVI": float(metadata["Pre_GSHH_NDVI"]),
            "Height_Ave_cm": float(metadata["Height_Ave_cm"]),
            "Ground Sampling Area (m²)": float(sampling_area),
            "Dry_Clover_g": float(predictions["Dry_Clover_g"]),
            "Dry_Dead_g": float(predictions["Dry_Dead_g"]),
            "Dry_Green_g": float(predictions["Dry_Green_g"]),
            "Dry_Total_g": float(predictions["Dry_Total_g"]),
            "GDM_g": float(predictions["GDM_g"]),
            "Dry_Total_kg_ha": float(density_predictions["Dry_Total_g"]),
            "GDM_kg_ha": float(density_predictions["GDM_g"]),
            "prediction_event_token": prediction_event_token,
        }
    )


    st.session_state["monitoring_last_committed_token"] = (
        prediction_event_token
    )


def load_prediction_image(image_reference, active_source):

    if active_source == "upload":

        return load_visual_image(
            image_reference,
            target_size=VISUAL_IMAGE_SIZE,
        )

    return load_visual_image(
        Path(image_reference),
        target_size=VISUAL_IMAGE_SIZE,
    )


# ============================================================
# SIDEBAR
# ============================================================

def navigation_callback():

    st.session_state[
        "current_page"
    ] = st.session_state[
        "navigation_widget"
    ]


def render_sidebar():

    with st.sidebar:

        st.title(
            "🌱 Image2Biomass"
        )


        st.caption(
            "Precision-agriculture "
            "biomass estimation"
        )


        st.radio(
            "Navigation",

            options=
                NAV_PAGES,

            key=
                "navigation_widget",

            on_change=
                navigation_callback,

            label_visibility=
                "collapsed",
        )


        st.divider()


        st.subheader(
            "Pipeline Status"
        )


        try:

            validation = (
                validate_resources()
            )


            configuration = (
                validation[
                    "configuration"
                ]
            )


            st.success(
                "Inference pipeline ready"
            )


            st.write(
                f"**Model:** "
                f"{configuration['model']}"
            )


            st.write(
                f"**Feature set:** "
                f"{configuration['feature_set']}"
            )


            st.write(
                f"**Features:** "
                f"{validation['n_canonical_features']}"
            )


            st.write(
                f"**Fold models:** "
                f"{validation['n_models']}"
            )


        except Exception as error:

            st.error(
                "Unable to validate "
                f"inference artifacts: {error}"
            )

            st.stop()


# ============================================================
# HERO
# ============================================================

def render_hero():

    st.markdown(
        """
<div class="hero-box">
    <div class="hero-title">Image2Biomass</div>
    <div class="hero-subtitle">
        Estimate pasture biomass from top-view RGB imagery and field metadata
        using computer vision and ensemble machine learning.
    </div>
    <span class="badge">Computer Vision</span>
    <span class="badge">Machine Learning</span>
    <span class="badge">Precision Agriculture</span>
</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOME PAGE
# ============================================================

def render_home_page():

    render_hero()


    with st.expander(
        "Image capture and input guidance"
    ):

        st.markdown(
            """
            For more reliable predictions:

            - Use a top-view RGB pasture image.
            - Keep vegetation clearly visible.
            - Avoid severe motion blur.
            - Avoid unrelated objects dominating the frame.
            - Use reasonably even natural lighting.
            - Provide field metadata associated with the image.

            Images substantially different from the training
            distribution may still produce numerical predictions,
            but those estimates may be unreliable.
            """
        )


    st.header(
        "Prediction Input"
    )


    st.markdown(
        """
        <div class="section-description">
            Upload a pasture image or select a real dataset sample,
            then provide the associated field information.
        </div>
        """,
        unsafe_allow_html=True,
    )


    image_column, metadata_column = (
        st.columns(
            [1, 1],
            gap="large",
        )
    )


    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    with image_column:

        st.subheader(
            "Pasture Image"
        )


        uploaded_image = (
            st.file_uploader(
                "Upload image",

                type=[
                    "jpg",
                    "jpeg",
                    "png",
                ],

                key=
                    "image_uploader",
            )
        )


        active_source = None

        image_reference = None

        image_caption = None


        if uploaded_image is not None:

            active_source = "upload"

            image_reference = uploaded_image

            image_caption = uploaded_image.name


            st.image(
                uploaded_image,

                caption=
                    uploaded_image.name,

                use_container_width=
                    True,
            )


            st.caption(
                "Active source: uploaded image"
            )


        else:

            selected_sample = (
                get_selected_sample()
            )


            if selected_sample is not None:

                active_source = "sample"

                image_reference = (
                    selected_sample[
                        "image_path"
                    ]
                )

                image_caption = (
                    selected_sample[
                        "name"
                    ]
                )


                if Path(
                    image_reference
                ).exists():

                    st.image(
                        str(
                            image_reference
                        ),

                        caption=
                            image_caption,

                        use_container_width=
                            True,
                    )


                    st.caption(
                        "Active source: "
                        "training dataset sample"
                    )


                    if st.button(
                        "Clear selected sample",

                        use_container_width=
                            True,
                    ):

                        clear_selected_sample()

                        st.rerun()


            else:

                st.info(
                    "Upload an image here or "
                    "choose a sample from the "
                    "Try a Sample page."
                )


    # --------------------------------------------------------
    # METADATA INPUT
    # --------------------------------------------------------

    with metadata_column:

        st.subheader(
            "Field Metadata"
        )


        st.selectbox(
            "State",

            options=
                VALID_STATES,

            key=
                "metadata_state",
        )


        st.selectbox(
            "Species",

            options=
                SPECIES_OPTIONS,

            key=
                "metadata_species",
        )


        st.number_input(
            "Pre-GSHH NDVI",

            min_value=
                0.0,

            max_value=
                1.0,

            step=
                0.01,

            format=
                "%.4f",

            key=
                "metadata_ndvi",
        )


        st.number_input(
            "Average Height (cm)",

            min_value=
                0.0,

            step=
                0.1,

            format=
                "%.4f",

            key=
                "metadata_height",
        )


        st.divider()


        st.subheader(
            "Ground Sampling Area"
        )


        st.number_input(
            "Sampling Area (m²)",

            min_value=
                0.0001,

            step=
                0.1,

            format=
                "%.4f",

            key=
                "sampling_area",

            help=
                (
                    "Enter the actual ground area "
                    "represented by the biomass sample."
                ),
        )


        st.caption(
            "kg/ha = "
            "(Predicted grams ÷ Area in m²) × 10"
        )


    # --------------------------------------------------------
    # INPUT REVIEW
    # --------------------------------------------------------

    with st.expander(
        "Review prediction inputs"
    ):

        review_1, review_2 = (
            st.columns(2)
        )


        with review_1:

            if active_source == "upload":

                source_text = (
                    "Uploaded image"
                )

            elif active_source == "sample":

                source_text = (
                    "Training sample"
                )

            else:

                source_text = (
                    "No image selected"
                )


            st.write(
                "**Image source:**",
                source_text,
            )


            st.write(
                "**State:**",
                st.session_state[
                    "metadata_state"
                ],
            )


            st.write(
                "**Species:**",
                st.session_state[
                    "metadata_species"
                ],
            )


        with review_2:

            st.write(
                "**NDVI:**",
                (
                    f"{st.session_state['metadata_ndvi']:.4f}"
                ),
            )


            st.write(
                "**Average height:**",
                (
                    f"{st.session_state['metadata_height']:.4f} cm"
                ),
            )


            st.write(
                "**Sampling area:**",
                (
                    f"{st.session_state['sampling_area']:.4f} m²"
                ),
            )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predict_button = st.button(
        "Predict Biomass",

        type=
            "primary",

        use_container_width=
            True,
    )


    if predict_button:

        run_prediction(

            active_source=
                active_source,

            image_reference=
                image_reference,

            image_caption=
                image_caption,
        )


    if (
        st.session_state[
            "last_predictions"
        ]
        is not None
    ):

        render_prediction_results()


# ============================================================
# PREDICTION EXECUTION
# ============================================================

def run_prediction(
    active_source,
    image_reference,
    image_caption,
):

    if active_source is None:

        st.warning(
            "Please upload an image "
            "or select a sample."
        )

        return


    metadata = {

        "State":
            st.session_state[
                "metadata_state"
            ],

        "Species":
            st.session_state[
                "metadata_species"
            ],

        "Pre_GSHH_NDVI":
            st.session_state[
                "metadata_ndvi"
            ],

        "Height_Ave_cm":
            st.session_state[
                "metadata_height"
            ],
    }


    sampling_area = float(
        st.session_state[
            "sampling_area"
        ]
    )


    temporary_path = None


    try:

        if active_source == "upload":

            suffix = Path(
                image_reference.name
            ).suffix


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    image_reference.getbuffer()
                )


                temporary_path = Path(
                    temp_file.name
                )


            inference_image_path = (
                temporary_path
            )


        else:

            inference_image_path = Path(
                image_reference
            )


            if not inference_image_path.exists():

                raise FileNotFoundError(
                    "Sample image does not exist: "
                    f"{inference_image_path}"
                )


        with st.spinner(
            "Extracting features and "
            "estimating pasture biomass..."
        ):

            predictions = (
                predict_single_sample(

                    image_path=
                        inference_image_path,

                    metadata=
                        metadata,
                )
            )


        density_predictions = (
            calculate_kg_ha(
                predictions=
                    predictions,

                sampling_area_m2=
                    sampling_area,
            )
        )


        st.session_state[
            "last_predictions"
        ] = predictions


        st.session_state[
            "last_density_predictions"
        ] = density_predictions


        st.session_state[
            "last_sampling_area"
        ] = sampling_area


        st.session_state[
            "last_image_caption"
        ] = image_caption


        try:

            visualization_image = load_prediction_image(
                inference_image_path,
                active_source,
            )


            visualization_artifacts = (
                build_vegetation_biomass_proxy_heatmap(
                    visualization_image
                )
            )


            st.session_state[
                "last_vegetation_source_image"
            ] = visualization_image


            st.session_state[
                "last_vegetation_heatmap"
            ] = visualization_artifacts["heatmap"]


            st.session_state[
                "last_vegetation_overlay"
            ] = visualization_artifacts["overlay"]


            st.session_state[
                "last_vegetation_heatmap_error"
            ] = None


        except Exception as heatmap_error:

            st.session_state[
                "last_vegetation_source_image"
            ] = None


            st.session_state[
                "last_vegetation_heatmap"
            ] = None


            st.session_state[
                "last_vegetation_overlay"
            ] = None


            st.session_state[
                "last_vegetation_heatmap_error"
            ] = str(heatmap_error)


        prediction_event_token = uuid.uuid4().hex


        append_monitoring_record(
            image_source_name=image_caption,
            image_source_type=(
                "Uploaded Image"
                if active_source == "upload"
                else "Dataset Sample"
            ),
            metadata=metadata,
            sampling_area=sampling_area,
            predictions=predictions,
            density_predictions=density_predictions,
            prediction_event_token=prediction_event_token,
        )


        st.success(
            "Biomass prediction completed successfully."
        )


    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            temporary_path.unlink()


# ============================================================
# RESULTS
# ============================================================

def render_prediction_results():

    predictions = (
        st.session_state[
            "last_predictions"
        ]
    )


    density_predictions = (
        st.session_state[
            "last_density_predictions"
        ]
    )


    sampling_area = (
        st.session_state[
            "last_sampling_area"
        ]
    )


    st.divider()


    st.header(
        "Prediction Results"
    )


    if st.session_state[
        "last_image_caption"
    ]:

        st.caption(
            "Prediction source: "
            f"{st.session_state['last_image_caption']}"
        )


    st.subheader(
        "Predicted Biomass"
    )


    gram_columns = st.columns(
        len(TARGET_COLUMNS)
    )


    for column, target in zip(
        gram_columns,
        TARGET_COLUMNS,
    ):

        column.metric(

            TARGET_DISPLAY_LABELS.get(
                target,
                target,
            ),

            f"{float(predictions[target]):.2f} g",
        )


    st.subheader(
        "Estimated Biomass Density"
    )


    density_columns = st.columns(
        len(TARGET_COLUMNS)
    )


    for column, target in zip(
        density_columns,
        TARGET_COLUMNS,
    ):

        column.metric(

            TARGET_DISPLAY_LABELS.get(
                target,
                target,
            ),

            (
                f"{float(density_predictions[target]):.2f} "
                "kg/ha"
            ),
        )


    results_dataframe = pd.DataFrame(
        {
            "Biomass Component": [

                TARGET_DISPLAY_LABELS.get(
                    target,
                    target,
                )

                for target
                in TARGET_COLUMNS
            ],

            "Predicted Biomass (g)": [

                float(
                    predictions[target]
                )

                for target
                in TARGET_COLUMNS
            ],

            "Estimated Density (kg/ha)": [

                float(
                    density_predictions[target]
                )

                for target
                in TARGET_COLUMNS
            ],
        }
    )


    st.subheader(
        "Detailed Results"
    )


    st.dataframe(
        results_dataframe,

        use_container_width=
            True,

        hide_index=
            True,
    )


    chart_1, chart_2 = (
        st.columns(
            2,
            gap="large",
        )
    )


    with chart_1:

        st.subheader(
            "Biomass Composition"
        )


        st.bar_chart(

            results_dataframe[
                [
                    "Biomass Component",
                    "Predicted Biomass (g)",
                ]
            ]
            .set_index(
                "Biomass Component"
            )
        )


    with chart_2:

        st.subheader(
            "Estimated Density"
        )


        st.bar_chart(

            results_dataframe[
                [
                    "Biomass Component",
                    "Estimated Density (kg/ha)",
                ]
            ]
            .set_index(
                "Biomass Component"
            )
        )


    with st.expander(
        "Conversion details"
    ):

        st.write(
            "**Ground sampling area:** "
            f"{sampling_area:.4f} m²"
        )


        st.write(
            "Biomass density is calculated using:"
        )


        st.markdown(
            """
            **kg/ha = (Predicted grams ÷ Ground Sampling Area in m²) × 10**
            """
        )


        st.caption(
            "The density estimate depends on the accuracy "
            "of the ground sampling area supplied by the user."
        )


    st.divider()

    render_vegetation_proxy_heatmap_section()


    st.info(
        "Predictions are model estimates. Images and metadata "
        "substantially different from the training distribution "
        "may produce less reliable estimates."
    )


def render_vegetation_proxy_heatmap_section():

    st.subheader("Vegetation Biomass Proxy Heatmap")

    st.caption(
        "The heatmap visualizes relative vegetation intensity derived "
        "from image characteristics. It is a visual proxy and does not "
        "represent pixel-level biomass predictions from the regression model."
    )

    heatmap_error = st.session_state.get("last_vegetation_heatmap_error")

    if heatmap_error:

        st.warning(
            "Heatmap generation was skipped for this assessment: "
            f"{heatmap_error}"
        )

        return

    source_image = st.session_state.get("last_vegetation_source_image")
    heatmap_image = st.session_state.get("last_vegetation_heatmap")

    if source_image is None or heatmap_image is None:

        st.info(
            "The proxy heatmap appears after a successful prediction."
        )

        return

    heatmap_columns = st.columns(
        2,
        gap="large",
    )

    with heatmap_columns[0]:

        st.image(
            source_image,
            caption="Original pasture image",
            use_container_width=True,
        )


    with heatmap_columns[1]:

        st.image(
            heatmap_image,
            caption="Vegetation Biomass Proxy Heatmap",
            use_container_width=True,
        )


    overlay_image = st.session_state.get("last_vegetation_overlay")

    if overlay_image is not None:

        with st.expander("Optional overlay view"):

            st.image(
                overlay_image,
                caption="Overlay of relative vegetation intensity on the pasture image",
                use_container_width=True,
            )


# ============================================================
# TRY SAMPLE PAGE
# ============================================================

def render_sample_page():

    st.title("Try a Sample")

    st.caption(
        "Explore valid pasture samples from the training dataset. "
        "Selecting a sample automatically transfers its image and "
        "metadata to the prediction interface."
    )

    sample_error = st.session_state.get(
        "sample_selection_error"
    )

    if sample_error:

        st.error(sample_error)

        st.session_state["sample_selection_error"] = None

    if not SAMPLE_CONFIG:

        st.warning(
            "No valid dataset samples are currently available."
        )

        return

    sample_columns = st.columns(
        3,
        gap="large",
    )

    for index, sample in enumerate(SAMPLE_CONFIG):

        column = sample_columns[index % 3]

        with column:

            with st.container(border=True):

                image_path = Path(sample["image_path"])

                if not image_path.exists():

                    st.warning("Sample image unavailable.")

                    continue

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

                st.subheader(sample["name"])

                st.write(
                    f"**State:** {sample['state']}"
                )

                st.write(
                    f"**Species:** "
                    f"{sample['species'].replace('_', ' ')}"
                )

                st.write(
                    f"**NDVI:** {sample['ndvi']:.4f}"
                )

                st.write(
                    f"**Average Height:** "
                    f"{sample['height_cm']:.4f} cm"
                )

                if st.button(
                    "Use This Sample",
                    key=f"use_sample_{index}",
                    type="primary",
                    use_container_width=True,
                ):

                    select_sample(sample)

                    st.rerun()


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

def render_model_page():

    st.title("Model Information")

    st.caption(
        "Production inference architecture and model "
        "configuration used by Image2Biomass."
    )

    validation = validate_resources()

    configuration = validation["configuration"]

    metric_columns = st.columns(
        4,
        gap="large",
    )

    metric_columns[0].metric(
        "Production Model",
        configuration["model"],
    )

    metric_columns[1].metric(
        "Feature Configuration",
        configuration["feature_set"],
    )

    metric_columns[2].metric(
        "Predictive Features",
        validation["n_canonical_features"],
    )

    metric_columns[3].metric(
        "Cross-Validation Models",
        validation["n_models"],
    )

    st.divider()

    st.header("Prediction Pipeline")

    st.markdown(
        """
<div class="pipeline-container">
<div class="pipeline-step">
<div class="pipeline-step-title">Pasture Image</div>
<div class="pipeline-step-description">Top-view RGB pasture imagery</div>
</div>
<div class="pipeline-arrow">↓</div>
<div class="pipeline-step">
<div class="pipeline-step-title">EfficientNetB0 CNN Features</div>
<div class="pipeline-step-description">Deep visual representation extracted from the pasture image</div>
</div>
<div class="pipeline-arrow">+</div>
<div class="pipeline-step">
<div class="pipeline-step-title">Field Metadata Features</div>
<div class="pipeline-step-description">State, species, NDVI and average pasture height</div>
</div>
<div class="pipeline-arrow">+</div>
<div class="pipeline-step">
<div class="pipeline-step-title">Handcrafted Visual Features</div>
<div class="pipeline-step-description">Additional image and vegetation characteristics</div>
</div>
<div class="pipeline-arrow">↓</div>
<div class="pipeline-step">
<div class="pipeline-step-title">1389 Predictive Features</div>
<div class="pipeline-step-description">Canonical combined production feature representation</div>
</div>
<div class="pipeline-arrow">↓</div>
<div class="pipeline-step">
<div class="pipeline-step-title">Five ExtraTrees Fold Models</div>
<div class="pipeline-step-description">Cross-validation ensemble regression models</div>
</div>
<div class="pipeline-arrow">↓</div>
<div class="pipeline-step">
<div class="pipeline-step-title">Fold Averaging</div>
<div class="pipeline-step-description">Predictions are averaged across validated fold models</div>
</div>
<div class="pipeline-arrow">↓</div>
<div class="pipeline-step">
<div class="pipeline-step-title">Biomass Estimation</div>
<div class="pipeline-step-description">Five biomass predictions in grams with optional kg/ha conversion</div>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.header("Prediction Targets")

    target_dataframe = pd.DataFrame(
        {
            "Technical Target": TARGET_COLUMNS,

            "Display Name": [
                TARGET_DISPLAY_LABELS.get(
                    target,
                    target,
                )
                for target in TARGET_COLUMNS
            ],
        }
    )

    st.dataframe(
        target_dataframe,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ABOUT PAGE
# ============================================================

def render_about_page():

    st.title("About Image2Biomass")

    st.caption(
        "Computer vision and ensemble machine learning "
        "for pasture biomass estimation."
    )

    card_1, card_2, card_3 = st.columns(
        3,
        gap="large",
    )

    with card_1:

        with st.container(border=True):

            st.subheader("The Challenge")

            st.write(
                "Traditional pasture biomass measurement can "
                "require harvesting, drying and weighing vegetation, "
                "making frequent field assessment labor intensive."
            )

    with card_2:

        with st.container(border=True):

            st.subheader("The Approach")

            st.write(
                "Image2Biomass combines deep CNN embeddings, "
                "field metadata and handcrafted visual features "
                "with ensemble machine-learning regression."
            )

    with card_3:

        with st.container(border=True):

            st.subheader("The Output")

            st.write(
                "The system estimates five pasture biomass "
                "components in grams and supports ground-area-based "
                "conversion to estimated kilograms per hectare."
            )

    st.divider()

    st.header("Biomass Components")

    st.markdown(
        """
- **Dry Clover** — estimated dry clover biomass.
- **Dry Dead Material** — estimated dry dead pasture biomass.
- **Dry Green Vegetation** — estimated dry green vegetation.
- **Total Dry Biomass** — estimated total dry biomass.
- **Green Dry Matter** — estimated green dry matter.
        """
    )

    st.header("Important Limitations")

    st.warning(
        "The model was trained on a specific pasture-image "
        "dataset and associated field conditions. Images or "
        "metadata substantially different from the training "
        "distribution may produce unreliable estimates. "
        "The kg/ha conversion also depends directly on the "
        "accuracy of the supplied ground sampling area."
    )


def render_monitoring_dashboard():

    st.title("Monitoring Dashboard")

    st.caption(
        "This dashboard records timestamped on-demand pasture assessments "
        "during the current Streamlit session. It does not represent a "
        "continuous live sensor feed."
    )

    history = pd.DataFrame(
        st.session_state.get("monitoring_history", [])
    )

    if history.empty:

        st.info(
            "No monitoring records have been captured yet. Run a successful "
            "prediction to populate this dashboard."
        )

    else:

        state_options = ["All States"] + sorted(
            history["State"].dropna().astype(str).unique().tolist()
        )

        species_options = ["All Species"] + sorted(
            history["Species"].dropna().astype(str).unique().tolist()
        )

        filter_columns = st.columns(2)

        with filter_columns[0]:

            selected_state = st.selectbox(
                "Filter by State",
                options=state_options,
                key="monitoring_filter_state",
            )


        with filter_columns[1]:

            selected_species = st.selectbox(
                "Filter by Species",
                options=species_options,
                key="monitoring_filter_species",
            )


        filtered_history = history.copy()

        if selected_state != "All States":

            filtered_history = filtered_history[
                filtered_history["State"] == selected_state
            ]


        if selected_species != "All Species":

            filtered_history = filtered_history[
                filtered_history["Species"] == selected_species
            ]


        filtered_history = filtered_history.copy()
        filtered_history["Timestamp_dt"] = pd.to_datetime(
            filtered_history["Timestamp"],
            errors="coerce",
        )
        filtered_history = filtered_history.sort_values(
            "Timestamp_dt",
            ascending=False,
            na_position="last",
        )

        if filtered_history.empty:

            st.warning(
                "No monitoring records match the current filters."
            )

        else:

            summary_columns = st.columns(4)

            summary_columns[0].metric(
                "Total Assessments",
                f"{len(filtered_history)}",
            )

            summary_columns[1].metric(
                "Latest Total Dry Biomass",
                f"{float(filtered_history.iloc[0]['Dry_Total_g']):.2f} g",
            )

            summary_columns[2].metric(
                "Latest GDM",
                f"{float(filtered_history.iloc[0]['GDM_g']):.2f} g",
            )

            summary_columns[3].metric(
                "Average Total Dry Biomass",
                f"{float(filtered_history['Dry_Total_g'].mean()):.2f} g",
            )

            st.subheader("Monitoring History")

            display_history = filtered_history.drop(
                columns=[
                    "prediction_event_token",
                    "Timestamp_dt",
                ],
                errors="ignore",
            )

            st.dataframe(
                display_history,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Biomass Trend")

            if len(filtered_history) < 2:

                st.info(
                    "At least two assessments are required to display a trend chart."
                )

            else:

                trend_frame = filtered_history.sort_values(
                    "Timestamp_dt",
                    ascending=True,
                ).set_index("Timestamp_dt")

                st.line_chart(
                    trend_frame[
                        ["Dry_Total_g", "GDM_g"]
                    ]
                )

            st.subheader("Biomass Density Trend")

            if len(filtered_history) < 2:

                st.info(
                    "At least two assessments are required to display a density trend chart."
                )

            else:

                density_frame = filtered_history.sort_values(
                    "Timestamp_dt",
                    ascending=True,
                ).set_index("Timestamp_dt")

                st.line_chart(
                    density_frame[
                        ["Dry_Total_kg_ha", "GDM_kg_ha"]
                    ]
                )

        st.divider()

        if st.session_state.get("monitoring_clear_confirmation"):

            st.warning(
                "Confirming this action will permanently clear the session monitoring history."
            )

            clear_columns = st.columns(2)

            with clear_columns[0]:

                if st.button(
                    "Confirm Clear History",
                    type="primary",
                    use_container_width=True,
                    key="monitoring_clear_confirm_button",
                ):

                    st.session_state["monitoring_history"] = []
                    st.session_state["monitoring_clear_confirmation"] = False
                    st.session_state["monitoring_last_committed_token"] = None
                    st.rerun()


            with clear_columns[1]:

                if st.button(
                    "Cancel",
                    use_container_width=True,
                    key="monitoring_clear_cancel_button",
                ):

                    st.session_state["monitoring_clear_confirmation"] = False
                    st.rerun()


        elif st.button(
            "Clear Monitoring History",
            use_container_width=True,
            key="monitoring_clear_request_button",
        ):

            st.session_state["monitoring_clear_confirmation"] = True
            st.rerun()

# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.divider()


    st.caption(
        "Image2Biomass · Computer Vision and Machine Learning "
        "for Pasture Biomass Estimation"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    init_session_state()

    # --------------------------------------------------------
    # PROCESS PENDING NAVIGATION
    # --------------------------------------------------------

    pending_navigation = st.session_state.get(
        "pending_navigation"
    )

    if pending_navigation is not None:

        st.session_state["current_page"] = pending_navigation

        st.session_state["navigation_widget"] = pending_navigation

        st.session_state["pending_navigation"] = None

    render_custom_css()

    render_sidebar()

    current_page = st.session_state["current_page"]

    if current_page == "Home / Prediction":

        render_home_page()

    elif current_page == "Try a Sample":

        render_sample_page()

    elif current_page == "Monitoring Dashboard":

        render_monitoring_dashboard()

    elif current_page == "Model Information":

        render_model_page()

    elif current_page == "About the Project":

        render_about_page()

    render_footer()


if __name__ == "__main__":
    main()