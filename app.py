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

from tensorflow.keras.models import load_model

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

# ─── Load Models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    lstm  = load_model('models/model_lstm.keras', compile=False)
    rf    = joblib.load('models/model_rf.joblib')
    xgb   = joblib.load('models/model_xgb.joblib')
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
    data          = data[:, :MIN_FEATURES]
    # fit fresh scaler on uploaded data itself
    fresh_scaler  = MinMaxScaler()
    data_scaled   = fresh_scaler.fit_transform(data)
    windows       = create_windows(data_scaled, WINDOW_SIZE, STEP)
    flat_features = extract_features(windows)
    return windows, flat_features

def predict_all(windows, flat_features):
    # XGBoost
    xgb_pred  = model_xgb.predict(flat_features)
    xgb_prob  = model_xgb.predict_proba(flat_features)[:, 1]

    # Random Forest
    rf_pred   = model_rf.predict(flat_features)
    rf_prob   = model_rf.predict_proba(flat_features)[:, 1]

    # LSTM
    lstm_prob = model_lstm.predict(windows, verbose=0).flatten()
    lstm_pred = (lstm_prob > LSTM_THRESHOLD).astype(int)

    return xgb_pred, xgb_prob, rf_pred, rf_prob, lstm_pred, lstm_prob

# ─── Sidebar ──────────────────────────────────────────────
st.sidebar.title("🛸 NASA Anomaly Detection")
st.sidebar.markdown("**Dataset:** SMAP & MSL Telemetry")
st.sidebar.markdown("**Models:** XGBoost | Random Forest | BiLSTM")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Predict", "📊 Model Comparison", "🧠 SHAP Explanation"])

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
    This project detects anomalies in NASA spacecraft telemetry data using three machine learning models:
    - **XGBoost** - Gradient boosting on 125 statistical features (Winner 🏆)
    - **Random Forest** - Ensemble of 100 decision trees
    - **BiLSTM + Attention** - Deep learning on raw telemetry sequences
    
    The dataset contains telemetry from two NASA spacecraft:
    - **SMAP** - Soil Moisture Active Passive satellite
    - **MSL** - Mars Science Laboratory (Curiosity Rover)
    """)

    st.markdown("---")
    st.markdown("### Model Performance Summary")
    results_df = pd.DataFrame({
        'Model'    : ['XGBoost 🏆', 'Random Forest', 'BiLSTM + Attention'],
        'Accuracy' : ['96.21%', '82.00%', '~52.43%'],
        'F1 Score' : ['86.17%', '55.42%', '~31.72%'],
        'ROC-AUC'  : ['98.77%', '93.72%', '73.01%']
    })

    results_df.index = results_df.index + 1
    
    st.dataframe(results_df, use_container_width=True)

# ─── Page 2: Predict ──────────────────────────────────────
elif page == "🔍 Predict":
    st.title("🔍 Anomaly Prediction")
    st.markdown("Upload a `.npy` telemetry file to detect anomalies using XGBoost.")
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
            with st.spinner("Running XGBoost prediction..."):
                windows, flat_features = preprocess(data)
                xgb_pred  = model_xgb.predict(flat_features)
                xgb_prob  = model_xgb.predict_proba(flat_features)[:, 1]

            total     = len(xgb_pred)
            anomalies = xgb_pred.sum()
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
                        true_window_labels = build_window_labels(true_labels, len(xgb_pred))
                        
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

            # map predictions to timesteps
            pred_signal = np.zeros(len(signal))
            for i, pred in enumerate(xgb_pred):
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
                axes[1].set_title(f'Channel {chan_name} - XGBoost Prediction')
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
                'Window'    : range(len(xgb_pred)),
                'Start'     : [i * STEP for i in range(len(xgb_pred))],
                'End'       : [i * STEP + WINDOW_SIZE for i in range(len(xgb_pred))],
                'Prediction': ['Anomaly' if p == 1 else 'Normal' for p in xgb_pred],
                'Confidence': [round(p * 100, 2) for p in xgb_prob]
            }).to_csv(index=False)

            st.download_button(
                label     = "📥 Download Predictions CSV",
                data      = csv,
                file_name = f"{chan_name}_predictions.csv",
                mime      = "text/csv"
            )

# ─── Page 3: Model Comparison ─────────────────────────────
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Upload a `.npy` file to compare all 3 models side by side.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload .npy telemetry file", type=['npy'])

    if uploaded_file is not None:
        data = np.load(uploaded_file)
        st.success(f"File loaded - Shape: {data.shape}")

        with st.spinner("Running all 3 models..."):
            windows, flat_features        = preprocess(data)
            xgb_pred, xgb_prob, rf_pred, rf_prob, lstm_pred, lstm_prob = predict_all(windows, flat_features)

        st.markdown("---")
        st.markdown("### Prediction Counts")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### XGBoost 🏆")
            st.metric("Normal",  (xgb_pred==0).sum())
            st.metric("Anomaly", (xgb_pred==1).sum())

        with col2:
            st.markdown("#### Random Forest")
            st.metric("Normal",  (rf_pred==0).sum())
            st.metric("Anomaly", (rf_pred==1).sum())

        with col3:
            st.markdown("#### BiLSTM")
            st.metric("Normal",  (lstm_pred==0).sum())
            st.metric("Anomaly", (lstm_pred==1).sum())

        st.markdown("---")
        st.markdown("### Anomaly Detection Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        models    = ['XGBoost', 'Random Forest', 'BiLSTM']
        anomalies = [(xgb_pred==1).sum(), (rf_pred==1).sum(), (lstm_pred==1).sum()]
        colors    = ['steelblue', 'darkorange', 'crimson']
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

        n_plots = 4 if has_ground_truth else 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, n_plots * 3))

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

        # model plots
        for ax, (name, pred, color) in zip(model_axes, [
            ('XGBoost',       xgb_pred,  'crimson'),
            ('Random Forest', rf_pred,   'darkorange'),
            ('BiLSTM',        lstm_pred, 'purple')
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

        plt.suptitle(f'Anomaly Timeline - {chan_name}', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)

# ─── Page 4: SHAP ─────────────────────────────────────────
elif page == "🧠 SHAP Explanation":
    st.title("🧠 SHAP Feature Explanation")
    st.markdown("Understand which telemetry features drive XGBoost anomaly predictions.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload .npy telemetry file", type=['npy'])

    if uploaded_file is not None:
        data = np.load(uploaded_file)
        st.success(f"File loaded — Shape: {data.shape}")

        with st.spinner("Computing SHAP values..."):
            windows, flat_features = preprocess(data)
            sample                 = flat_features[:200]
            explainer              = shap.TreeExplainer(model_xgb)
            shap_values            = explainer.shap_values(sample)

        st.markdown("### SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, max_display=15, show=False)
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Top 10 Most Important Features")
        feature_importance = pd.DataFrame({
            'Feature'   : [f'feature_{i}' for i in range(flat_features.shape[1])],
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=feature_importance, x='Importance', y='Feature',
                   palette='Blues_r', ax=ax)
        ax.set_title('Top 10 Features by SHAP Importance')
        st.pyplot(fig)
