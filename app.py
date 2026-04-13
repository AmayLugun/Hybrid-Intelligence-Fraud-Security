# app.py — Streamlit Dashboard for Fraud Detection
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model as keras_load_model

from src.preprocess import load_data, preprocess_data
from src.random_forest_model import train_random_forest, get_feature_importances
from src.neural_network import build_neural_network, train_neural_network, predict_neural_network
from src.kmeans import train_kmeans, evaluate_kmeans
from src.evaluation import evaluate_model, get_roc_curve, get_pr_curve
from src.visualization import (
    plot_heatmap, plot_feature_importance, plot_top_correlations, plot_class_distribution
)

# ── Paths & Config ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
RF_PATH, NN_PATH, SCALER_PATH = [os.path.join(BASE_DIR, "models", f) for f in ["rf_model.pkl", "nn_model.keras", "scaler.pkl"]]
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🔒", layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔒 Fraud Detection")
page = st.sidebar.radio("Navigation", ["📊 Data Explorer", "🤖 Train Models", "📈 Results", "🔍 Single Transaction"])
st.sidebar.markdown("---")
use_smote = st.sidebar.checkbox("Apply SMOTE", value=True)
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2)
rf_trees = st.sidebar.slider("RF: Trees", 50, 300, 100, 50)
nn_epochs = st.sidebar.slider("NN: Epochs", 3, 20, 10)

# ── State Initialization ───────────────────────────────────────────────────────
STATE_KEYS = ["df", "X_train", "X_test", "y_train", "y_test", "rf_model", "nn_model", "feature_names", "scaler", "rf_metrics", "nn_metrics", "rf_y_prob", "nn_y_prob"]
for key in STATE_KEYS:
    if key not in st.session_state: st.session_state[key] = None

if "init" not in st.session_state:
    if os.path.exists(DATA_PATH): 
        st.session_state.df = load_data(DATA_PATH)
        st.session_state.feature_names = np.array(st.session_state.df.drop("Class", axis=1).columns.tolist())
    if os.path.exists(RF_PATH): st.session_state.rf_model = joblib.load(RF_PATH)
    if os.path.exists(NN_PATH): st.session_state.nn_model = keras_load_model(NN_PATH)
    if os.path.exists(SCALER_PATH): st.session_state.scaler = joblib.load(SCALER_PATH)
    st.session_state.init = True

# ── Helpers ────────────────────────────────────────────────────────────────────
def run_preprocessing():
    X_tr, X_te, y_tr, y_te, sc = preprocess_data(st.session_state.df, use_smote=use_smote, test_size=test_size)
    st.session_state.update({"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te, "scaler": sc})
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(sc, SCALER_PATH)

def eval_and_store(name, model, X, y):
    if name == "RF":
        prob = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
    else:
        prob = predict_neural_network(model, X).flatten()
        pred = (prob > 0.5).astype(int)
    metrics = evaluate_model(y, pred, prob)
    st.session_state[f"{name.lower()}_metrics"] = metrics
    st.session_state[f"{name.lower()}_y_prob"] = prob
    return metrics

def status_indicator(loaded, label):
    if loaded: st.success(f"✅ {label} Loaded")
    else: st.warning(f"❌ No {label}")

# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up: 
        st.session_state.df = pd.read_csv(up)
        st.session_state.feature_names = np.array(st.session_state.df.drop("Class", axis=1).columns.tolist())
    
    df = st.session_state.df
    if df is not None:
        fraud = int(df.Class.sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"{len(df):,}")
        c2.metric("Legit", f"{len(df)-fraud:,}")
        c3.metric("Fraud", f"{fraud:,} ({fraud/len(df)*100:.2f}%)")

        m1, m2, m3 = st.columns(3)
        with m1: status_indicator(st.session_state.rf_model is not None, "RF Model")
        with m2: status_indicator(st.session_state.nn_model is not None, "NN Model")
        with m3: status_indicator(st.session_state.scaler is not None, "Scaler")

        tab_list = st.tabs(["Preview", "Stats", "Correlations", "Distribution", "Heatmap"])
        with tab_list[0]: st.dataframe(df.head(10), use_container_width=True)
        with tab_list[1]: st.dataframe(df.describe(), use_container_width=True)
        with tab_list[2]: st.pyplot(plot_top_correlations(df))
        with tab_list[3]:
            feat = st.selectbox("Feature", st.session_state.feature_names, index=list(st.session_state.feature_names).index('V17') if 'V17' in st.session_state.feature_names else 0)
            st.pyplot(plot_class_distribution(df, feat))
        with tab_list[4]: st.pyplot(plot_heatmap(df))
    else: st.info("Upload data to begin.")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.title("🤖 Training & Evaluation")
    if st.session_state.df is None: st.warning("No data."); st.stop()

    if st.session_state.X_test is None:
        if st.button("⚙️ Preprocess Data", use_container_width=True):
            run_preprocessing(); st.rerun()
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌲 Random Forest")
        b1, b2 = st.columns(2)
        if b1.button("📂 Load RF"): 
            if os.path.exists(RF_PATH): st.session_state.rf_model = joblib.load(RF_PATH); st.rerun()
        if b2.button("🔨 Train RF"):
            with st.spinner("Training RF..."):
                rf = train_random_forest(st.session_state.X_train, st.session_state.y_train, n_estimators=rf_trees)
                st.session_state.rf_model = rf
                joblib.dump(rf, RF_PATH)
                eval_and_store("RF", rf, st.session_state.X_test, st.session_state.y_test)
        if st.session_state.rf_model:
            m = st.session_state.rf_metrics or eval_and_store("RF", st.session_state.rf_model, st.session_state.X_test, st.session_state.y_test)
            st.write(f"Recall: {m['recall']:.4f} | AUC: {m['auc_roc']:.4f}")

    with col2:
        st.subheader("🧠 Neural Network")
        b3, b4 = st.columns(2)
        if b3.button("📂 Load NN"): 
            if os.path.exists(NN_PATH): st.session_state.nn_model = keras_load_model(NN_PATH); st.rerun()
        if b4.button("🔨 Train NN"):
            with st.spinner("Training NN..."):
                nn = train_neural_network(build_neural_network(st.session_state.X_train.shape[1]), st.session_state.X_train, st.session_state.y_train, epochs=nn_epochs)
                st.session_state.nn_model = nn
                nn.save(NN_PATH)
                eval_and_store("NN", nn, st.session_state.X_test, st.session_state.y_test)
        if st.session_state.nn_model:
            m = st.session_state.nn_metrics or eval_and_store("NN", st.session_state.nn_model, st.session_state.X_test, st.session_state.y_test)
            st.write(f"Recall: {m['recall']:.4f} | AUC: {m['auc_roc']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results":
    st.title("📈 Model Comparison")
    rf_m, nn_m = st.session_state.rf_metrics, st.session_state.nn_metrics
    if not rf_m and not nn_m: st.warning("No metrics."); st.stop()

    res = []
    if rf_m: res.append({"Model": "RF", **{k.capitalize(): round(rf_m[k], 4) for k in ["recall", "precision", "f1", "auc_roc"]}})
    if nn_m: res.append({"Model": "NN", **{k.capitalize(): round(nn_m[k], 4) for k in ["recall", "precision", "f1", "auc_roc"]}})
    st.table(pd.DataFrame(res).set_index("Model"))

    fig, ax = plt.subplots(figsize=(8, 4))
    if rf_m: 
        fpr, tpr, _ = get_roc_curve(st.session_state.y_test, st.session_state.rf_y_prob)
        ax.plot(fpr, tpr, label=f"RF (AUC={rf_m['auc_roc']:.3f})")
    if nn_m:
        fpr, tpr, _ = get_roc_curve(st.session_state.y_test, st.session_state.nn_y_prob)
        ax.plot(fpr, tpr, label=f"NN (AUC={nn_m['auc_roc']:.3f})")
    ax.plot([0,1],[0,1],'k--'); ax.legend(); st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Transaction":
    st.title("🔍 Prediction")
    if not st.session_state.rf_model and not st.session_state.nn_model: st.warning("Train models first."); st.stop()
    
    with st.form("input"):
        inputs = [st.number_input(f, value=0.0) for f in st.session_state.feature_names]
        if st.form_submit_button("Predict"):
            row = np.array([inputs])
            if st.session_state.scaler: row = st.session_state.scaler.transform(row)
            
            if st.session_state.rf_model:
                p = st.session_state.rf_model.predict_proba(row)[0][1]
                st.write(f"RF Fraud Prob: {p:.4f}")
            if st.session_state.nn_model:
                p = float(predict_neural_network(st.session_state.nn_model, row))
                st.write(f"NN Fraud Prob: {p:.4f}")