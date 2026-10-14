# NASA Telemetry Classification

A NASA spacecraft telemetry anomaly detection project that combines machine learning and explainable AI to detect abnormal patterns in telemetry streams from the SMAP and MSL missions. The app is built with Streamlit and uses an ensemble of XGBoost, Random Forest, and a BiLSTM-based deep learning model to classify anomaly windows with a majority-vote strategy.

This project is especially useful for interview conversations because it demonstrates:
- end-to-end ML pipeline design
- time-series anomaly detection
- model ensembling
- deployment in a user-facing app
- explainability using SHAP
- working with real scientific telemetry data

## 1. Project Objective

The goal is to detect abnormal behavior in spacecraft telemetry data before it causes mission-critical failures.

In practical terms, the model looks at a stream of sensor readings and tries to answer:
- Is this telemetry window normal or anomalous?
- Which regions of the signal look suspicious?
- Which features are driving the model's decision?

This is a classic time-series anomaly detection problem. The data is not just random noise; it is structured telemetry from real spacecraft systems, where anomalies can be subtle, contextual, or sudden.

## 2. Why This Project Matters

Space missions generate huge amounts of telemetry every second. NASA teams cannot manually inspect all of it.

This project helps by automating anomaly detection so engineers can:
- identify potential problems faster
- focus attention on suspicious telemetry windows
- reduce the risk of unnoticed system degradation
- improve operational monitoring of flight systems

From an interview perspective, this shows that you can work on a realistic, high-impact ML problem where the challenge is not only model accuracy, but also real-world interpretability and deployment.

## 3. Dataset Overview

The project uses telemetry data from NASA's:
- SMAP: Soil Moisture Active Passive
- MSL: Mars Science Laboratory (Curiosity rover)

The repository includes a labeled anomaly file:
- `labeled_anomalies.csv`

This file stores anomaly segments for each telemetry channel, including:
- `chan_id`: channel name
- `spacecraft`: mission source (SMAP or MSL)
- `anomaly_sequences`: list of anomalous ranges in the signal
- `class`: type of anomaly, such as point or contextual
- `num_values`: signal length

The dataset contains a mixture of:
- point anomalies: sudden short spikes or dips
- contextual anomalies: values that are not necessarily extreme but are abnormal relative to the surrounding signal

This is a very important concept in anomaly detection because many simple threshold-based methods fail when behavior is context-dependent.

## 4. Business / Technical Problem Framing

The core problem is: detect anomalies in multi-feature telemetry sequences without relying on hand-coded rules.

This type of problem is difficult because:
- signal patterns vary by spacecraft channel
- anomalies may be sparse but high-impact
- different channels have different behaviors
- contextual anomalies are not always easy to detect via simple statistics
- a single model may work well for some channels but fail for others

The project solves this by combining three models in an ensemble and using window-based detection over time series data.

## 5. Model Architecture

The app implements a majority-vote ensemble using three models:

1. Tuned XGBoost
   - uses engineered statistical features extracted from each telemetry window
   - very strong at tabular feature-based anomaly detection

2. Tuned Random Forest
   - another tree-based model trained on the same feature representation
   - good robustness and interpretability

3. Tuned BiLSTM with Attention
   - works on raw sequential telemetry windows
   - captures temporal dependencies and sequence patterns
   - more suitable for dynamic time-series behavior

Final decision:
- each model predicts whether a window is anomalous
- the ensemble label is the majority vote across the three models
- this reduces the risk of relying on one model's bias

This is a strong interview story: instead of using a single classifier, you combine models to improve stability and generalization.

## 6. Key Technical Details from the Code

The project uses these major design choices:

### 6.1 Windowing
The app defines:
- `WINDOW_SIZE = 128`
- `STEP = 10`

This means a telemetry stream is broken into overlapping windows of 128 timesteps, moved forward by 10 steps each time.

Why windowing matters:
- anomaly detection is often performed over temporal neighborhoods, not single points
- context around a suspicious event matters
- short local fluctuations can be evaluated in a richer sequence context

### 6.2 Feature Extraction
The `extract_features()` function computes statistics for each window:
- mean
- standard deviation
- min
- max
- range

This turns a time-series sequence into a compact feature vector that tree-based models can understand easily.

### 6.3 Scaling
The project uses `MinMaxScaler` to normalize telemetry data before model input.

This is important because:
- different telemetry channels may have different scales
- feature magnitude differences could bias the model
- scaling improves convergence and consistency for neural networks and tree models alike

### 6.4 Thresholding for LSTM
`LSTM_THRESHOLD = 0.4`

The BiLSTM model outputs a probability between 0 and 1. The project marks a window anomalous if the probability exceeds 0.4.

This means the model is not just giving a hard class label; it is also giving calibrated uncertainty representation.

### 6.5 SHAP Explainability
The Streamlit app includes a SHAP explanation page for the XGBoost model.

This is valuable because anomaly detection models are often black boxes. SHAP helps answer:
- which features contributed most to the anomaly decision?
- are the predictions driven by the right signals?
- can I explain the model to non-technical stakeholders?

This is a very good interview talking point because it shows you understand model trust and explainability.

## 7. Repository Structure

Here is the project layout:

```text
NASA-Telemetry-Classification/
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   └── config.toml
├── models/
│   ├── .gitattributes
│   ├── scaler.joblib
│   ├── tuned_lstm_scikeras.joblib
│   ├── tuned_rf_model.joblib
│   └── tuned_xgb_model.joblib
├── NASA_Telemetry_Anomaly_Classification.ipynb
├── README.md
├── app.py
├── labeled_anomalies.csv
├── requirements.txt
└── .gitattributes
```

## 8. File-by-File Explanation

### `app.py`
This is the heart of the project.

It contains:
- Streamlit UI definition
- model loading from `.joblib` files
- preprocessing functions
- window creation and feature engineering
- prediction functions for XGBoost, Random Forest, and LSTM
- majority-vote ensemble logic
- dashboards for anomaly prediction and comparison
- SHAP plots

This file is the main deliverable because it takes the trained models and turns them into an interactive app that can be used by someone without a coding background.

It also illustrates that the project is not just a notebook experiment; it is a working product-like interface.

### `NASA_Telemetry_Anomaly_Classification.ipynb`
This is the modeling notebook where the training and experimentation probably happened.

It is important because it documents the exploratory ML workflow:
- data loading
- preprocessing
- model training
- performance evaluation
- tuning
- validation

Interview angle:
- shows that you moved from experimentation to deployment
- demonstrates an understanding of model lifecycle and iterative experimentation

### `labeled_anomalies.csv`
This file provides ground-truth anomaly intervals for each telemetry channel.

It acts as the labeling reference for evaluation. Without it, you would not know whether the model is actually finding real anomalies.

This is crucial because anomaly detection often suffers from weak labels or ambiguous data. This project includes concrete labeled segments, which makes model validation much more rigorous.

### `models/scaler.joblib`
This stores the fitted `MinMaxScaler` used to normalize the telemetry data during inference.

This ensures the app uses the same scaling logic as the training pipeline.

### `models/tuned_xgb_model.joblib`
Trained XGBoost model for tabular anomaly detection using engineered window-level features.

This is one of the strongest performers in the project and is also used for SHAP explanations.

### `models/tuned_rf_model.joblib`
Trained Random Forest model on the same feature engineering pipeline.

Random Forest is often a very good baseline because it handles nonlinear relationships well and is resilient to noisy features.

### `models/tuned_lstm_scikeras.joblib`
Saved BiLSTM model wrapped with Scikeras/legacy Keras compatibility.

This is the deep learning component that learns temporal patterns directly from raw windows.

The project carefully handles compatibility since TensorFlow/Keras and SciKeras loading can be tricky. This is a strong sign of practical engineering understanding.

### `requirements.txt`
This lists the project dependencies:
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- TensorFlow CPU
- Joblib
- SHAP
- scikeras

This shows the project combines classical ML and deep learning within a single environment.

### `.devcontainer/devcontainer.json`
This defines the dev environment in a container.

It ensures the project can run consistently in:
- GitHub Codespaces
- VS Code dev containers
- reproducible developer environments

This helps with collaboration and reduces the "works on my machine" problem.

### `.streamlit/config.toml`
This configures Streamlit behavior, including:
- disabled CORS
- disabled XSRF protection
- max upload size

This is relevant when the app accepts `.npy` telemetry uploads in the browser.

### `.gitattributes`
This repository uses Git attributes to configure file handling. It is not the main logic, but it helps maintain repository consistency and large-file behavior.

## 9. How the System Works End-to-End

The typical pipeline is:

1. User uploads a `.npy` file containing telemetry data
2. Data is truncated to the required feature count
3. MinMaxScaler normalizes the selected features
4. The telemetry sequence is split into windows
5. Statistical features are extracted from each window
6. XGBoost and Random Forest predict anomaly labels on those features
7. LSTM processes the sequence windows and outputs a probability score
8. Majority voting combines the three individual predictions
9. The app visualizes anomaly regions over time
10. SHAP explains the XGBoost decision for interpretability

This is a strong end-to-end story for interviews because it demonstrates real ML pipeline thinking, not just model training in a notebook.

## 10. Model Performance Summary

The app reports the following performance summary:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Ensemble (Voting) | 98% | 92.59% |
| Tuned XGBoost | 98% | 97.43% |
| Tuned Random Forest | 98% | 97.41% |
| Tuned BiLSTM | 82% | 75.90% |

This is a good interview talking point because it shows:
- the project went beyond a simple prototype
- model comparison was performed
- ensembling was used to combine strengths
- performance was measured with meaningful classification metrics

## 11. Why This Project Is Strong for Interviews

This repository is interview-friendly because it demonstrates multiple skills at once:

- Time-series data analysis
- Feature engineering
- Model selection and comparison
- Ensemble learning
- Deep learning for sequences
- Explainable AI
- Deployment through Streamlit
- Real-world scientific use case

A strong answer in an interview could be:

> I built a spacecraft telemetry anomaly detection system using NASA SMAP and MSL data. The project processes telemetry windows, extracts statistical features, trains an XGBoost model and Random Forest on those features, and uses a BiLSTM with attention to model sequential dependencies. I combine these predictions with a majority-vote ensemble and deploy the system in a Streamlit app. I also use SHAP to explain which features contributed to each anomaly decision, making the model more transparent and useful for real-world monitoring.

## 12. How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

### Access the app
Open the generated localhost link, typically:

```text
http://localhost:8501
```

## 13. Business-Level Explanation for Interviews

If you need a simplified explanation in 30–60 seconds, you can say:

> This project detects anomalies in NASA spacecraft telemetry using machine learning. Instead of manually inspecting millions of sensor readings, I built a system that analyzes time-series windows, identifies abnormal behavior, and flags suspicious regions. The model uses a mix of tree-based models and a deep learning sequence model, then combines them through ensemble voting to improve robustness. I also added SHAP-based explainability so you can understand which telemetry features drove the anomaly detection decision.

## 14. Technical-Level Explanation for Interviews

> I developed a multistage anomaly detection pipeline for NASA telemetry. The system breaks telemetry streams into overlapping windows, extracts statistical features like mean, variance, and signal range, then trains XGBoost and Random Forest models on those engineered features. For sequential dependency modeling, I include a BiLSTM with attention on raw windows. The final prediction uses a majority-vote ensemble across the three models, and I use SHAP to interpret the XGBoost component. The project is packaged in a Streamlit app to allow interactive anomaly prediction and model comparison.

## 15. Final Summary

This repository is a strong example of applied ML because it combines:
- realistic scientific data
- time-series modeling
- ensemble techniques
- deep learning and classical ML
- deployment in an interactive interface
- explainability and model interpretation

It is a great project to discuss in interviews because it shows you can handle both model-building and product-style deployment, and that you understand the full workflow from raw telemetry to interpretable, actionable decisions.

---

If you want, I can also create a second version of this README tailored specifically for:
1. GitHub portfolio style
2. Interview answer style
3. recruiter-friendly summary
4. technical deep-dive version

I can also generate a polished one-page project pitch for your LinkedIn / GitHub profile.
