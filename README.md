# 🌱 Image2Biomass

## Automated Pasture Biomass Estimation from Field Images

Image2Biomass is an end-to-end computer vision and machine learning system for estimating pasture biomass from top-view RGB field images and associated field metadata.

The project combines deep visual representations extracted using EfficientNetB0, structured field metadata, and handcrafted vegetation and texture features with an ExtraTrees regression ensemble.

A Streamlit application provides interactive biomass estimation, biomass density conversion, vegetation proxy heatmap visualization, and session-based pasture monitoring.

---

## 📌 Project Overview

Accurate pasture biomass estimation is important for grazing management, feed planning, precision agriculture, and sustainable farming.

Traditional biomass measurement can require manual field sampling, vegetation harvesting, drying, and weighing. These processes are time-consuming and labor-intensive.

Image2Biomass explores an automated alternative using:

- RGB pasture imagery
- Deep learning-based visual feature extraction
- Field metadata
- Vegetation and image-processing features
- Ensemble machine learning
- Interactive model inference

The complete project covers data exploration, preprocessing, feature engineering, classical machine learning, deep learning experiments, final model comparison, production inference, and application development.

---

## 🎯 Expected Outcomes

The project is designed to support:

- Automated biomass estimation from field images
- On-demand and near-real-time pasture monitoring
- Vegetation biomass proxy heatmap visualization
- Improved grazing and feed planning
- Reduced labor and field survey requirements
- Precision agriculture and sustainable farming practices

---

## 📊 Dataset

The project uses the CSIRO Image2Biomass pasture dataset.

The training data contains:

- 357 unique pasture images
- Top-view RGB imagery
- Images with dimensions of `2000 × 1000 × 3`
- Four field metadata variables
- Five biomass regression targets

### Field Metadata

| Feature | Description |
|---|---|
| `State` | Australian state associated with the pasture sample |
| `Species` | Pasture species or species composition |
| `Pre_GSHH_NDVI` | Pre-measurement vegetation index |
| `Height_Ave_cm` | Average pasture height in centimeters |

### Prediction Targets

| Target | Description |
|---|---|
| `Dry_Clover_g` | Dry clover biomass |
| `Dry_Dead_g` | Dry dead vegetation biomass |
| `Dry_Green_g` | Dry green vegetation biomass |
| `Dry_Total_g` | Total dry biomass |
| `GDM_g` | Green dry matter |

---

## 🏗️ System Architecture

The final prediction pipeline combines three major feature groups:

1. EfficientNetB0 CNN features
2. Processed field metadata
3. Handcrafted visual features

```text
                         Pasture RGB Image
                                |
                +---------------+---------------+
                |                               |
                v                               v
       EfficientNetB0                    Visual Feature
       Feature Extraction                  Extraction
                |                               |
                |                      +--------+--------+
                |                      |                 |
                |                Vegetation         Texture /
                |                 Features          Image Features
                |                      |                 |
                +----------------------+-----------------+
                                       |
                                       v
                              Field Metadata
                                       |
                                       v
                            Metadata Preprocessing
                                       |
                                       v
                          1389 Predictive Features
                                       |
                                       v
                       Five ExtraTrees Fold Models
                                       |
                                       v
                              Fold Averaging
                                       |
                                       v
                         Five Biomass Predictions
```

---

# 🔬 Machine Learning Pipeline

## 1. Data Exploration

The dataset was inspected for:

- Dataset dimensions
- Unique pasture images
- Target distributions
- State distribution
- Species distribution
- Missing values
- Duplicate records
- Negative numerical values
- Missing images
- Corrupted images
- Image dimensions

The exploration stage confirmed that the source dataset was suitable for subsequent preprocessing and modeling.

---

## 2. Data Preprocessing

The preprocessing pipeline performs:

- Dataset validation
- Image path validation
- Metadata preparation
- Group-aware cross-validation fold creation
- Fold assignment generation
- Metadata preprocessing
- Reusable preprocessing artifact generation

Five folds are used throughout the project to maintain consistent model evaluation.

---

## 3. Deep Feature Extraction

EfficientNetB0 is used as a pretrained convolutional neural network feature extractor.

The network converts pasture images into compact deep visual representations.

```text
Pasture Image
      |
      v
EfficientNetB0
      |
      v
Global Feature Representation
      |
      v
1280 CNN Features
```

The extracted embeddings are cached to prevent repeated CNN computation during model experimentation.

---

## 4. Baseline Modeling

Initial regression experiments establish classical machine learning baselines.

The baseline stage evaluates the predictive capability of the extracted image and metadata features before introducing additional feature engineering and advanced models.

Cross-validation predictions and evaluation reports are generated for model comparison.

---

## 5. Visual Features and Segmentation

Additional handcrafted image features are extracted to represent pasture characteristics not explicitly captured by structured metadata.

These include vegetation, color, texture, and edge-based characteristics.

Examples include:

- Excess Green vegetation characteristics
- Brown vegetation proxies
- Vegetation coverage features
- Color statistics
- Texture features
- Edge density

The visual feature pipeline also supports the application's vegetation biomass proxy heatmap.

---

## 6. Advanced Modeling

Advanced machine learning experiments combine:

- 1280 EfficientNetB0 CNN features
- Processed metadata features
- Handcrafted visual features

Multiple feature configurations and regression approaches are evaluated using five-fold cross-validation.

The final selected classical model is:

**ExtraTrees Regressor with All Features**

---

## 7. Deep Learning Regression

A separate deep learning regression pipeline was developed and evaluated.

The deep learning workflow includes:

- Fold-based model training
- Checkpoint generation
- Training history tracking
- Out-of-fold prediction generation
- Learning curve visualization
- Residual analysis
- Per-target evaluation

The deep learning models were retained as experimental results but were not selected as the final production inference strategy.

---

## 8. Final Model Comparison and Inference

The final stage compares:

- Baseline machine learning
- Advanced classical machine learning
- Deep learning regression
- Ensemble strategies

The final evaluation selected the advanced ExtraTrees model as the production strategy.

```text
Classical Model Weight = 1.0
CNN Regression Weight  = 0.0
```

Therefore, only the validated ExtraTrees ensemble is required for production inference.

---

# 📈 Final Model Performance

The selected production configuration achieved:

| Metric | Score |
|---|---:|
| Mean MAE | 8.0941 |
| Mean RMSE | 11.8411 |
| Mean R² | 0.6303 |
| Cross-Validation Folds | 5 |
| Predictive Features | 1389 |

## Model Comparison

![Model Comparison](assets/model_comparison.png)

## Actual vs Predicted Biomass

![Actual vs Predicted](assets/actual_vs_predicted.png)

---

# ⚙️ Production Inference Architecture

Training and experimental artifacts are separated from the production application.

The deployment pipeline uses a dedicated artifact directory:

```text
production_artifacts/
|
+---models/
|   +--- 5 ExtraTrees fold models
|
+---metadata_preprocessors/
|   +--- 5 fitted metadata preprocessors
|
+---configuration/
    +--- Best model configuration
    +--- Final inference strategy
    +--- Canonical feature column order
```

The production pipeline requires:

- 5 ExtraTrees fold models
- 5 metadata preprocessors
- 1389 canonical feature names
- Best-model configuration
- Final inference strategy

This design eliminates the requirement to load the complete training feature table during application inference.

---

# ✅ Inference Validation

The production inference module performs artifact validation before prediction.

Validation confirms:

```text
Canonical Features       : 1389
ExtraTrees Fold Models   : 5
Metadata Preprocessors   : 5
Prediction Targets       : 5
```

A consistency test was also performed between the notebook inference implementation and the reusable production inference module.

The test produced:

```text
Maximum Absolute Difference: 0.0000007487
Mean Absolute Difference   : 0.0000003414
```

The production inference module successfully passed consistency validation.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

The application provides five main pages:

```text
Home / Prediction
Try a Sample
Monitoring Dashboard
Model Information
About the Project
```

---

## 🌿 Biomass Prediction

Users can:

- Upload a pasture image
- Select a real dataset sample
- Enter or automatically populate field metadata
- Run production model inference
- View five biomass predictions
- Convert biomass estimates to kg/ha
- Inspect biomass composition charts

The application supports the following metadata inputs:

- State
- Species
- Pre-GSHH NDVI
- Average pasture height
- Ground sampling area

---

## 🖼️ Dataset Sample Selection

The application includes valid pasture samples from the training dataset.

Selecting a sample automatically transfers:

- Pasture image
- State
- Species
- NDVI
- Average pasture height

to the prediction interface.

Only samples with valid image files and complete required metadata are displayed.

---

## 🔥 Vegetation Biomass Proxy Heatmap

The application provides a vegetation intensity visualization derived from image characteristics.

The visualization compares:

- Original pasture image
- Vegetation biomass proxy heatmap

The heatmap provides a visual representation of relative vegetation intensity.

> **Important:** The heatmap is a vegetation proxy visualization and does not represent pixel-level biomass predictions from the ExtraTrees regression model.

---

## 📡 Pasture Monitoring Dashboard

The monitoring dashboard supports on-demand and near-real-time pasture assessment.

Every successful prediction can be recorded with:

- Timestamp
- Image source
- State
- Species
- NDVI
- Average pasture height
- Ground sampling area
- Five biomass predictions
- Total dry biomass density
- Green dry matter density

The dashboard provides:

- Total assessment count
- Latest biomass measurements
- Average biomass statistics
- Monitoring history
- Biomass trend visualization
- Biomass density trends
- State and species filtering
- Session history management

Monitoring history is maintained using Streamlit session state.

The system is designed for interactive on-demand monitoring and does not claim continuous live drone, camera, or IoT monitoring.

---

## 📐 Biomass Density Conversion

The application supports conversion from predicted biomass in grams to estimated biomass density in kilograms per hectare.

The conversion is:

```text
kg/ha = (Predicted Biomass in Grams / Sampling Area in m²) × 10
```

The reliability of this conversion depends on the accuracy of the supplied ground sampling area.

---

# 📁 Repository Structure

```text
Image2Biomass/
|
+---app/
|   +---app.py
|
+---assets/
|   +---08_model_comparison.png
|   +---08_best_strategy_actual_vs_predicted.png
|
+---notebooks/
|   +---01_data_exploration.ipynb
|   +---02_data_preprocessing.ipynb
|   +---03_feature_extraction.ipynb
|   +---04_baseline_modeling.ipynb
|   +---05_visual_features_and_segmentation.ipynb
|   +---06_advanced_modeling.ipynb
|   +---07_deep_learning_regression.ipynb
|   +---08_final_model_comparison_and_inference.ipynb
|
+---production_artifacts/
|   |
|   +---models/
|   |
|   +---metadata_preprocessors/
|   |
|   +---configuration/
|
+---src/
|   +---advanced_modeling.py
|   +---config.py
|   +---feature_extraction.py
|   +---inference.py
|   +---metadata_features.py
|   +---modeling.py
|   +---visual_features.py
|
+---.gitignore
+---requirements.txt
+---README.md
```

The following directories are generated locally and excluded from version control:

```text
data/raw/
data/processed/
models/
outputs/
```

---

# 📓 Notebook Workflow

The project should be explored in the following order:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Dataset inspection and exploratory analysis |
| `02_data_preprocessing.ipynb` | Data validation, metadata processing, and fold generation |
| `03_feature_extraction.ipynb` | EfficientNetB0 and metadata feature extraction |
| `04_baseline_modeling.ipynb` | Baseline machine learning experiments |
| `05_visual_features_and_segmentation.ipynb` | Visual feature extraction and vegetation analysis |
| `06_advanced_modeling.ipynb` | Advanced feature combinations and model evaluation |
| `07_deep_learning_regression.ipynb` | CNN-based regression experiments |
| `08_final_model_comparison_and_inference.ipynb` | Final comparison, strategy selection, and inference validation |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Image2Biomass
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

The project uses:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Scikit-image
- OpenCV
- Pillow
- Joblib
- TensorFlow
- Streamlit
- tqdm

---

# ▶️ Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The application validates the required production inference artifacts before accepting prediction requests.

---

# 📒 Running the Notebooks

The notebooks depend on the original dataset and generated intermediate artifacts.

To reproduce the complete machine learning pipeline, execute the notebooks sequentially:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

The original dataset should be placed under the appropriate `data/raw/` structure before running the notebooks.

---

# 🛠️ Technology Stack

## Programming and Data Processing

- Python
- NumPy
- Pandas

## Computer Vision

- OpenCV
- Scikit-image
- Pillow

## Machine Learning

- Scikit-learn
- ExtraTrees Regression
- Five-fold cross-validation

## Deep Learning

- TensorFlow
- Keras
- EfficientNetB0

## Visualization

- Matplotlib
- Seaborn

## Application

- Streamlit

## Artifact Management

- Joblib
- JSON configuration artifacts

---

# 💡 Key Design Decisions

## Group-Aware Evaluation

The project uses consistent fold assignments to reduce evaluation leakage and maintain comparable experiments.

## Cached Deep Features

EfficientNetB0 embeddings are extracted once and cached to avoid repeated CNN computation.

## Modular Source Code

Reusable project logic is separated into the `src/` package instead of placing all implementation directly inside notebooks.

## Experiment and Production Separation

Experimental models and training outputs are separated from the artifacts required for production inference.

## Lightweight Production Feature Schema

The canonical 1389-feature order is stored as a lightweight JSON artifact instead of requiring the complete training feature table during inference.

## Fold Ensemble

Predictions from five independently trained ExtraTrees fold models are averaged to generate the final biomass estimates.

---

# ⚠️ Limitations

The current system has several limitations:

- The training dataset contains a relatively limited number of unique pasture images.
- Predictions may be less reliable for images substantially different from the training distribution.
- Field metadata must be supplied accurately.
- kg/ha conversion depends on the correctness of the supplied sampling area.
- The vegetation heatmap is a visual proxy and not a pixel-level biomass prediction.
- The monitoring dashboard maintains session-based history rather than persistent database storage.
- The application supports on-demand inference rather than continuous live camera, drone, or IoT streaming.

---

# 🔮 Future Improvements

Potential future extensions include:

- Larger and more geographically diverse pasture datasets
- Additional environmental and weather features
- Persistent monitoring history using a database
- Geographic information system integration
- Drone imagery support
- Automated field-area estimation
- Temporal biomass forecasting
- Model uncertainty estimation
- Explainable AI techniques
- Model compression and inference optimization
- Cloud deployment
- Continuous monitoring through camera or IoT integration

---

# 🌍 Applications

Image2Biomass can support research and development related to:

- Precision agriculture
- Pasture management
- Grazing optimization
- Feed planning
- Agricultural monitoring
- Sustainable farming
- Computer vision for agriculture
- Automated field assessment

---

# 📌 Project Status

The implemented project includes:

- Data exploration
- Data preprocessing
- Cross-validation fold generation
- EfficientNetB0 feature extraction
- Metadata feature engineering
- Visual feature extraction
- Vegetation segmentation analysis
- Baseline modeling
- Advanced machine learning
- Deep learning regression experiments
- Final model comparison
- Production artifact generation
- Reusable inference module
- Inference consistency validation
- Interactive Streamlit application
- Vegetation biomass proxy heatmap
- Session-based monitoring dashboard
- Biomass density conversion

---

# ⚖️ Disclaimer

Image2Biomass is a machine learning project developed for pasture biomass estimation research and educational purposes.

Predictions should be interpreted as model estimates and should not replace professional agronomic assessment or validated field measurement procedures.

---

# 👨‍💻 Author

**Krish R Patel**
AI/ML Intern
DATA VIDWAN
Computer Engineering  
LDRP Institute of Technology and Research

---

# 🙏 Acknowledgements

This project uses the CSIRO Image2Biomass dataset and builds upon open-source Python, computer vision, machine learning, and deep learning libraries.