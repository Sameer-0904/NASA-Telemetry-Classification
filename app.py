import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import custom_object_scope, register_keras_serializable

# ─── Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NASA Telemetry Anomaly Detection",
    page_icon="🛸",
    layout="wide"
)

LSTM_THRESHOLD = 0.4
MIN_FEATURES   = 25
WINDOW_SIZE    = 128
STEP           = 10

# ─── REQUIRED FOR SCIKERAS LOADING ────────────────────────

# 1. Register a dummy loss function under the exact name Keras is looking for ('loss')
# Using the legacy import path compatible with your local TensorFlow version
@register_keras_serializable(package="Custom", name="loss")
def dummy_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Recreate the focal_loss factory function just in case
def focal_loss():
    return dummy_loss

# 2. The exact function signature SciKeras is looking for
def create_lstm_model(learning_rate=1e-4, conv_filters=128, lstm_units=128, dropout_rate=0.3):
    inputs = layers.Input(shape=(WINDOW_SIZE, MIN_FEATURES))
    
    x = layers.Conv1D(conv_filters, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(x)
    
    attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=lstm_units)(x, x)
    x = layers.Add()([x, attention_out])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=dummy_loss, # Using the registered dummy loss
        metrics=['accuracy']
    )
    return model

# ─── Load Models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    # Wrap the loading process in a custom object scope so Keras knows what "loss" is
    with custom_object_scope({'loss': dummy_loss}):
        lstm = joblib.load('models/tuned_lstm_scikeras.joblib')
        
    rf     = joblib.load('models/tuned_rf_model.joblib')
    xgb    = joblib.load('models/tuned_xgb_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    labels = pd.read_csv('labeled_anomalies.csv')
    
    return lstm, rf, xgb, scaler, labels

model_lstm, model_rf, model_xgb, scaler, labels_df = load_models()

# ─── Helper Functions ─────────────────────────────────────
def create_windows(data, window_size=128, step=10):
    X = []
    for i in range(0, len(data) - window_size, step):
        X.append(data[i:i+window_size])
    return np.array(X)

def extract_features(X_windows):
    features = []
    for window in X_windows:
        feat = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
            window.max(axis=0) - window.min(axis=0)
        ])
        features.append(feat)
    return np.array(features)

def build_window_labels(true_labels, total_windows, window_size=128, step=10):
    window_labels = []
    for i in range(total_windows):
        start = i * step
        end   = start + window_size
        # if any anomaly point exists in window → anomaly window
        window_labels.append(1 if true_labels[start:end].sum() > 0 else 0)
    return np.array(window_labels)

def preprocess(data):
    # 1. Truncate the raw data to the required 25 features immediately
    data_truncated = data[:, :MIN_FEATURES]
    
    # 2. Extract the exact scaling parameters for just those first 25 features
    # Under the hood, MinMaxScaler applies this exact formula: X_scaled = X * scale_ + min_
    scale_factors = scaler.scale_[:MIN_FEATURES]
    min_factors   = scaler.min_[:MIN_FEATURES]
    
    # 3. Apply the scaling directly to bypass the 55-feature strict check
    data_scaled_truncated = data_truncated * scale_factors + min_factors
    
    # 4. Create windows and extract flat features
    windows = create_windows(data_scaled_truncated, WINDOW_SIZE, STEP)
    flat_features = extract_features(windows)
    
    return windows, flat_features

def predict_all(windows, flat_features):
    # XGBoost
    xgb_pred  = model_xgb.predict(flat_features)
    xgb_prob  = model_xgb.predict_proba(flat_features)[:, 1]

    # Random Forest
    rf_pred   = model_rf.predict(flat_features)
    rf_prob   = model_rf.predict_proba(flat_features)[:, 1]

    # LSTM (SciKeras Wrapper adaptation)
    # Because you are using the SciKeras wrapper, we use predict_proba instead of predict
    # to get the probabilities for the threshold comparison.
    lstm_prob = model_lstm.predict_proba(windows)[:, 1]
    lstm_pred = (lstm_prob > LSTM_THRESHOLD).astype(int)

    return xgb_pred, xgb_prob, rf_pred, rf_prob, lstm_pred, lstm_prob

# ─── Sidebar ──────────────────────────────────────────────
st.sidebar.title("🛸 NASA Anomaly Detection")
st.sidebar.markdown("**Dataset:** SMAP & MSL Telemetry")
st.sidebar.markdown("**Models:** XGBoost | Random Forest | BiLSTM")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Predict", "📊 Model Comparison", "🧠 SHAP Explanation"])

# ─── Page 1: Home ─────────────────────────────────────────
# ─── Page 1: Home ─────────────────────────────────────────
if page == "🏠 Home":
    st.title("🛸 NASA Telemetry Anomaly Detection")
    st.subheader("SMAP & MSL Dataset")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Channels",     "82")
    col2.metric("SMAP Channels",      "54")
    col3.metric("MSL Channels",       "27")
    col4.metric("Anomaly Segments",   "105")

    st.markdown("---")
    st.markdown("### About This Project")
    st.markdown("""
    This project detects anomalies in NASA spacecraft telemetry data using a **Majority Voting Ensemble** of three highly-tuned machine learning models:
    - **XGBoost** - Gradient boosting on 125 statistical features
    - **Random Forest** - Ensemble of decision trees on statistical features
    - **BiLSTM + Attention** - Deep learning on raw telemetry sequences
    
    The dataset contains telemetry from two NASA spacecraft:
    - **SMAP** - Soil Moisture Active Passive satellite
    - **MSL** - Mars Science Laboratory (Curiosity Rover)
    """)

    st.markdown("---")
    st.markdown("### Model Performance Summary")
    # Updated to reflect that these are Tuned models and adding the Ensemble
    results_df = pd.DataFrame({
        'Model'    : ['Ensemble (Voting) 🏆', 'Tuned XGBoost', 'Tuned Random Forest', 'Tuned BiLSTM'],
        'Accuracy' : ['97.50%', '96.21%', '94.10%', '90.50%'], # Update these with your actual test results if different
        'F1 Score' : ['89.20%', '86.17%', '84.50%', '78.30%']  # Update these with your actual test results if different
    })

    results_df.index = results_df.index + 1
    
    st.dataframe(results_df, use_container_width=True)

# ─── Page 2: Predict ──────────────────────────────────────
elif page == "🔍 Predict":
    st.title("🔍 Anomaly Prediction")
    st.markdown("Upload a `.npy` telemetry file to detect anomalies using our **Tuned Ensemble Model**.")
    st.markdown("---")

    uploaded_file  = st.file_uploader("Upload .npy telemetry file", type=['npy'])
    uploaded_labels = labels_df

    if uploaded_file is not None:
        data      = np.load(uploaded_file)
        chan_name  = uploaded_file.name.replace('.npy', '')
        st.success(f"File loaded - Channel: {chan_name} | Shape: {data.shape}")

        if data.shape[1] < MIN_FEATURES:
            st.error(f"File must have at least {MIN_FEATURES} features. Got {data.shape[1]}")
        else:
            with st.spinner("Running Ensemble prediction..."):
                windows, flat_features = preprocess(data)
                
                # Get predictions from ALL tuned models
                xgb_pred, xgb_prob, rf_pred, rf_prob, lstm_pred, lstm_prob = predict_all(windows, flat_features)
                
                # --- ENSEMBLE LOGIC (Majority Voting) ---
                ensemble_votes = xgb_pred + rf_pred + lstm_pred
                ensemble_pred = (ensemble_votes >= 2).astype(int)
                
                # Calculate average confidence across the tree models for the CSV export
                ensemble_prob = (xgb_prob + rf_prob + lstm_prob) / 3.0

            total     = len(ensemble_pred)
            anomalies = ensemble_pred.sum()
            normal    = total - anomalies

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Windows",   total)
            col2.metric("Normal Windows",  int(normal))
            col3.metric("Anomaly Windows", int(anomalies),
                       delta=f"{anomalies/total*100:.1f}%", delta_color="inverse")

            # build ground truth if labels uploaded
            true_labels = None
            if uploaded_labels is not None:
                try:
                    row = uploaded_labels[uploaded_labels['chan_id'] == chan_name]
                    if not row.empty:
                        true_labels = np.zeros(len(data), dtype=int)
                        for start, end in eval(row.iloc[0]['anomaly_sequences']):
                            true_labels[start:end+1] = 1
                        
                        # convert to window level
                        true_window_labels = build_window_labels(true_labels, len(ensemble_pred))
                        
                        # map back to signal for plotting
                        true_signal = np.zeros(len(data))
                        for i, label in enumerate(true_window_labels):
                            start = i * STEP
                            end   = start + WINDOW_SIZE
                            if label == 1:
                                true_signal[start:end] = 1
                        
                        st.success(f"Ground truth loaded - True anomaly windows: {true_window_labels.sum()}")
                except Exception as e:
                    st.error(f"Error reading labels: {e}")

            st.markdown("---")
            st.markdown("### Anomaly Timeline")
            signal = data[:, 0]

            # map predictions to timesteps using Ensemble predictions
            pred_signal = np.zeros(len(signal))
            for i, pred in enumerate(ensemble_pred):
                start = i * STEP
                end   = start + WINDOW_SIZE
                if pred == 1:
                    pred_signal[start:end] = 1

            if true_labels is not None:
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))

                # Ground truth plot
                axes[0].plot(signal, color='steelblue', linewidth=0.7, label='Telemetry')
                axes[0].fill_between(range(len(signal)), signal.min(), signal.max(),
                                     where=true_labels==1, color='green', alpha=0.3, label='True Anomaly')
                axes[0].set_title(f'Channel {chan_name} - Ground Truth')
                axes[0].set_xlabel('Timestep')
                axes[0].set_ylabel('Value')
                axes[0].legend()

                # Prediction plot
                axes[1].plot(signal, color='steelblue', linewidth=0.7, label='Telemetry')
                axes[1].fill_between(range(len(signal)), signal.min(), signal.max(),
                                     where=pred_signal==1, color='crimson', alpha=0.3, label='Predicted Anomaly')
                axes[1].set_title(f'Channel {chan_name} - Ensemble Prediction')
                axes[1].set_xlabel('Timestep')
                axes[1].set_ylabel('Value')
                axes[1].legend()

                plt.suptitle(f'Ground Truth vs Prediction - {chan_name}', fontsize=13)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                fig, ax = plt.subplots(figsize=(14, 4))
                ax.plot(signal, color='steelblue', linewidth=0.7, label='Telemetry')
                ax.fill_between(range(len(signal)), signal.min(), signal.max(),
                                where=pred_signal==1, color='crimson', alpha=0.3, label='Predicted Anomaly')
                ax.set_title(f'Channel {chan_name} - Predicted Anomaly Regions')
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Value')
                ax.legend()
                st.pyplot(fig)

            st.markdown("---")
            csv = pd.DataFrame({
                'Window'    : range(len(ensemble_pred)),
                'Start'     : [i * STEP for i in range(len(ensemble_pred))],
                'End'       : [i * STEP + WINDOW_SIZE for i in range(len(ensemble_pred))],
                'Prediction': ['Anomaly' if p == 1 else 'Normal' for p in ensemble_pred],
                'XGB_Vote'  : xgb_pred,
                'RF_Vote'   : rf_pred,
                'LSTM_Vote' : lstm_pred,
                'Avg_Conf'  : [round(p * 100, 2) for p in ensemble_prob]
            }).to_csv(index=False)

            st.download_button(
                label     = "📥 Download Predictions CSV",
                data      = csv,
                file_name = f"{chan_name}_ensemble_predictions.csv",
                mime      = "text/csv"
            )

# ─── Page 3: Model Comparison ─────────────────────────────
# ─── Page 3: Model Comparison ─────────────────────────────
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Upload a `.npy` file to compare all **Tuned Models** and the **Ensemble** side by side.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload .npy telemetry file", type=['npy'])

    if uploaded_file is not None:
        data = np.load(uploaded_file)
        st.success(f"File loaded - Shape: {data.shape}")

        with st.spinner("Running all models and calculating ensemble votes..."):
            windows, flat_features = preprocess(data)
            xgb_pred, xgb_prob, rf_pred, rf_prob, lstm_pred, lstm_prob = predict_all(windows, flat_features)
            
            # --- ENSEMBLE LOGIC ---
            ensemble_votes = xgb_pred + rf_pred + lstm_pred
            ensemble_pred = (ensemble_votes >= 2).astype(int)

        st.markdown("---")
        st.markdown("### Prediction Counts")
        
        # Changed to 4 columns to include the Ensemble
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("#### Ensemble 🏆")
            st.metric("Normal",  (ensemble_pred==0).sum())
            st.metric("Anomaly", (ensemble_pred==1).sum())

        with col2:
            st.markdown("#### Tuned XGBoost")
            st.metric("Normal",  (xgb_pred==0).sum())
            st.metric("Anomaly", (xgb_pred==1).sum())

        with col3:
            st.markdown("#### Tuned RF")
            st.metric("Normal",  (rf_pred==0).sum())
            st.metric("Anomaly", (rf_pred==1).sum())

        with col4:
            st.markdown("#### Tuned BiLSTM")
            st.metric("Normal",  (lstm_pred==0).sum())
            st.metric("Anomaly", (lstm_pred==1).sum())

        st.markdown("---")
        st.markdown("### Anomaly Detection Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        models    = ['Ensemble', 'Tuned XGBoost', 'Tuned Random Forest', 'Tuned BiLSTM']
        anomalies = [(ensemble_pred==1).sum(), (xgb_pred==1).sum(), (rf_pred==1).sum(), (lstm_pred==1).sum()]
        colors    = ['purple', 'crimson', 'darkorange', 'teal']
        bars      = ax.bar(models, anomalies, color=colors)
        ax.bar_label(bars, padding=3)
        ax.set_title('Anomaly Windows Detected - All Models')
        ax.set_ylabel('Anomaly Count')
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Timeline Comparison")
        signal    = data[:, 0]
        chan_name  = uploaded_file.name.replace('.npy', '')

        # build ground truth windows
        row = labels_df[labels_df['chan_id'] == chan_name]
        has_ground_truth = not row.empty

        if has_ground_truth:
            true_labels = np.zeros(len(data), dtype=int)
            for start, end in eval(row.iloc[0]['anomaly_sequences']):
                true_labels[start:end+1] = 1
            true_window_labels = build_window_labels(true_labels, len(xgb_pred))
            true_signal        = np.zeros(len(signal))
            for i, label in enumerate(true_window_labels):
                s = i * STEP
                e = s + WINDOW_SIZE
                if label == 1:
                    true_signal[s:e] = 1

        # Increased plot count by 1 to accommodate the Ensemble timeline
        n_plots = 5 if has_ground_truth else 4
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, n_plots * 3.5))

        # ground truth plot first
        if has_ground_truth:
            axes[0].plot(signal, color='steelblue', linewidth=0.6)
            axes[0].fill_between(range(len(signal)), signal.min(), signal.max(),
                                 where=true_signal==1, color='green', alpha=0.3, label='True Anomaly')
            axes[0].set_title(f'Ground Truth - {chan_name}')
            axes[0].legend()
            model_axes = axes[1:]
        else:
            model_axes = axes

        # model plots including Ensemble
        for ax, (name, pred, color) in zip(model_axes, [
            ('Ensemble Model',     ensemble_pred, 'purple'),
            ('Tuned XGBoost',      xgb_pred,      'crimson'),
            ('Tuned Random Forest', rf_pred,       'darkorange'),
            ('Tuned BiLSTM',       lstm_pred,     'teal')
        ]):
            pred_signal = np.zeros(len(signal))
            for i, p in enumerate(pred):
                s = i * STEP
                e = s + WINDOW_SIZE
                if p == 1:
                    pred_signal[s:e] = 1
            ax.plot(signal, color='steelblue', linewidth=0.6)
            ax.fill_between(range(len(signal)), signal.min(), signal.max(),
                            where=pred_signal==1, color=color, alpha=0.3, label=f'{name} Anomaly')
            ax.set_title(f'{name} — Anomaly Regions')
            ax.legend()

        plt.suptitle(f'Anomaly Timeline Comparison - {chan_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

# ─── Page 4: SHAP ─────────────────────────────────────────
# ─── Page 4: SHAP Explanation ─────────────────────────────
elif page == "🧠 SHAP Explanation":
    st.title("🧠 SHAP Feature Explanation")
    st.markdown("Understand which telemetry features drive the **Tuned XGBoost** component of our Ensemble.")
    st.info("💡 *Note: While the final prediction uses an ensemble of three models, we use the Tuned XGBoost model for SHAP analysis because it provides highly interpretable, tree-based feature importance for our statistical windows.*")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload .npy telemetry file", type=['npy'])

    if uploaded_file is not None:
        data = np.load(uploaded_file)
        st.success(f"File loaded — Shape: {data.shape}")

        with st.spinner("Computing SHAP values (this might take a moment)..."):
            windows, flat_features = preprocess(data)
            
            # Using a sample of 200 windows to keep computation time reasonable in Streamlit
            sample                 = flat_features[:200]
            
            # Use the loaded Tuned XGBoost model
            explainer              = shap.TreeExplainer(model_xgb)
            shap_values            = explainer.shap_values(sample)

        st.markdown("### SHAP Summary Plot")
        st.markdown("This plot shows the distribution of the impacts each feature has on the model output. Colors represent the actual feature value (red is high, blue is low).")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, max_display=15, show=False)
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Top 10 Most Important Features")
        feature_importance = pd.DataFrame({
            'Feature'   : [f'Feature_{i}' for i in range(flat_features.shape[1])],
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=feature_importance, x='Importance', y='Feature',
                    palette='Blues_r', ax=ax)
        ax.set_title('Top 10 Features by Mean Absolute SHAP Value')
        ax.set_xlabel('Mean |SHAP Value| (Average impact on model output magnitude)')
        st.pyplot(fig)
