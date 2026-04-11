# app.py — Streamlit Dashboard: Hybrid Intelligence for Financial Security
#
# Run with:  streamlit run app.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model as keras_load_model

from src.preprocess import load_data, preprocess_data, split_data
from src.random_forest_model import train_random_forest, get_feature_importances, evaluate_random_forest
from src.neural_network import build_neural_network, train_neural_network, predict_neural_network, evaluate_neural_network
from src.kmeans import train_kmeans, evaluate_kmeans
from src.evaluation import evaluate_model, get_roc_curve, get_pr_curve
from src.visualization import plot_heatmap, plot_feature_importance

# ── Model Paths ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
NN_MODEL_PATH = os.path.join(BASE_DIR, "models", "nn_model.keras")
SCALER_PATH   = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH     = os.path.join(BASE_DIR, "data", "creditcard.csv")

# ── Cached Functions ───────────────────────────────────────────────────────────
@st.cache_data
def cached_heatmap(_df_hash, df):
    """Compute and return the correlation heatmap figure."""
    return plot_heatmap(df)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔒 Fraud Detection")
st.sidebar.markdown("**Hybrid Intelligence for Financial Security**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Data Explorer", "🤖 Train Models", "📈 Results", "🔍 Single Transaction"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
use_smote = st.sidebar.checkbox("Apply SMOTE", value=True)
test_size  = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
rf_trees   = st.sidebar.slider("RF: Number of Trees", 50, 300, 100, 50)
nn_epochs  = st.sidebar.slider("NN: Training Epochs", 3, 20, 10, 1)

# ── Session State ──────────────────────────────────────────────────────────────
for key in ["df", "X_train", "X_test", "y_train", "y_test",
            "rf_model", "nn_model", "feature_names", "scaler",
            "rf_metrics", "nn_metrics",
            "rf_y_pred", "rf_y_prob", "nn_y_pred", "nn_y_prob"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Auto-load pretrained models on first run ───────────────────────────────────
if "models_checked" not in st.session_state:
    st.session_state["models_checked"] = True

    # Auto-load dataset if it exists locally
    if os.path.exists(DATA_PATH) and st.session_state["df"] is None:
        st.session_state["df"] = load_data(DATA_PATH)
        st.session_state["feature_names"] = np.array(
            st.session_state["df"].drop("Class", axis=1).columns.tolist()
        )

    # Auto-load Random Forest
    if os.path.exists(RF_MODEL_PATH) and st.session_state["rf_model"] is None:
        st.session_state["rf_model"] = joblib.load(RF_MODEL_PATH)

    # Auto-load Neural Network
    if os.path.exists(NN_MODEL_PATH) and st.session_state["nn_model"] is None:
        st.session_state["nn_model"] = keras_load_model(NN_MODEL_PATH)

    # Auto-load Scaler
    if os.path.exists(SCALER_PATH) and st.session_state["scaler"] is None:
        st.session_state["scaler"] = joblib.load(SCALER_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — run preprocessing and store results
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocessing():
    df = st.session_state["df"]
    # New preprocess_data handles splitting internally to avoid leakage
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df, use_smote=use_smote, test_size=test_size
    )
    feature_names = np.array(df.drop("Class", axis=1).columns.tolist())
    st.session_state.update({
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
    })
    # Persist scaler to disk
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Explorer":
    st.title("📊 Data Explorer")

    uploaded = st.file_uploader("Upload creditcard.csv (or use auto-loaded local data)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.session_state["feature_names"] = np.array(
            df.drop("Class", axis=1).columns.tolist()
        )

    df = st.session_state["df"]

    if df is not None:
        fraud = int(df["Class"].sum())
        legit = len(df) - fraud

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Legitimate", f"{legit:,}")
        col3.metric("Fraud", f"{fraud:,}  ({fraud/len(df)*100:.3f}%)")

        # Show pretrained model status
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            if st.session_state["rf_model"] is not None:
                st.success("✅ Random Forest loaded")
            else:
                st.warning("❌ No RF model")
        with m2:
            if st.session_state["nn_model"] is not None:
                st.success("✅ Neural Network loaded")
            else:
                st.warning("❌ No NN model")
        with m3:
            if st.session_state["scaler"] is not None:
                st.success("✅ Scaler loaded")
            else:
                st.warning("❌ No scaler")

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Statistics", "Class Distribution", "Correlation Heatmap"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True)

        with tab2:
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)
            st.subheader("Missing Values")
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) == 0:
                st.success("No missing values found.")
            else:
                st.dataframe(missing.rename("Count"), use_container_width=True)

            st.subheader("Transaction Amount Distribution")
            if "Amount" in df.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(df["Amount"], bins=100, edgecolor="black", alpha=0.7)
                ax.set_xlabel("Amount")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Transaction Amount")
                st.pyplot(fig)

        with tab3:
            counts = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
            st.bar_chart(counts)

        with tab4:
            st.info("Computing heatmap — may take a moment for large datasets.")
            # Include Class in the heatmap to see feature correlations with the target
            fig_hm = cached_heatmap(hash(df.shape), df)
            st.pyplot(fig_hm)
            plt.close(fig_hm)
    else:
        st.info("👆 Upload creditcard.csv to get started, or place it at `data/creditcard.csv` for auto-loading.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Train Models
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.title("🤖 Train Models")

    if st.session_state["df"] is None:
        st.warning("Please upload a dataset on the **Data Explorer** page first.")
        st.stop()

    df = st.session_state["df"]

    # ── Preprocess button ────────────────────────────────────────────────────
    if st.session_state["X_test"] is None:
        st.info("💡 Preprocess data first before training or evaluating models.")
        if st.button("⚙️ Preprocess Data", use_container_width=True):
            with st.spinner("Preprocessing…"):
                run_preprocessing()
            st.success(f"Done! Train: {st.session_state['X_train'].shape[0]:,} | Test: {st.session_state['X_test'].shape[0]:,} | Scaler saved ✅")
            st.rerun()
        st.stop()
    else:
        st.success(f"Data preprocessed — Train: {st.session_state['X_train'].shape[0]:,} | Test: {st.session_state['X_test'].shape[0]:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Random Forest ────────────────────────────────────────────────────────
    with col1:
        st.subheader("🌲 Random Forest")

        if st.session_state["rf_model"] is not None:
            st.info("✅ RF model in memory")

        btn_col1, btn_col2 = st.columns(2)
        load_rf = btn_col1.button("📂 Load Saved", key="load_rf", use_container_width=True)
        train_rf = btn_col2.button("🔨 Train New", key="train_rf", use_container_width=True)

        if load_rf:
            if os.path.exists(RF_MODEL_PATH):
                st.session_state["rf_model"] = joblib.load(RF_MODEL_PATH)
                st.success("Loaded from `models/rf_model.pkl`")
                st.rerun()
            else:
                st.error(f"Model file not found: `{RF_MODEL_PATH}`")

        if train_rf:
            X_train = st.session_state["X_train"]
            X_test = st.session_state["X_test"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]

            with st.spinner(f"Training Random Forest ({rf_trees} trees)…"):
                rf = train_random_forest(X_train, y_train, n_estimators=rf_trees)

            # Save trained model
            os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
            joblib.dump(rf, RF_MODEL_PATH)
            st.session_state["rf_model"] = rf
            st.success("✅ Random Forest trained & saved!")

            # Evaluate
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_prob)
            st.session_state["rf_metrics"] = metrics
            st.session_state["rf_y_pred"] = y_pred
            st.session_state["rf_y_prob"] = y_prob

            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
            st.metric("F1", f"{metrics['f1']:.4f}")
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

        # Auto-evaluate if model loaded but no metrics yet
        if (st.session_state["rf_model"] is not None
                and st.session_state["X_test"] is not None
                and st.session_state["rf_metrics"] is None
                and not train_rf):
            rf = st.session_state["rf_model"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_prob)
            st.session_state["rf_metrics"] = metrics
            st.session_state["rf_y_pred"] = y_pred
            st.session_state["rf_y_prob"] = y_prob
            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

    # ── Neural Network ───────────────────────────────────────────────────────
    with col2:
        st.subheader("🧠 Neural Network")

        if st.session_state["nn_model"] is not None:
            st.info("✅ NN model in memory")

        btn_col3, btn_col4 = st.columns(2)
        load_nn = btn_col3.button("📂 Load Saved", key="load_nn", use_container_width=True)
        train_nn = btn_col4.button("🔨 Train New", key="train_nn", use_container_width=True)

        if load_nn:
            if os.path.exists(NN_MODEL_PATH):
                st.session_state["nn_model"] = keras_load_model(NN_MODEL_PATH)
                st.success("Loaded from `models/nn_model.keras`")
                st.rerun()
            else:
                st.error(f"Model file not found: `{NN_MODEL_PATH}`")

        if train_nn:
            X_train = st.session_state["X_train"]
            X_test = st.session_state["X_test"]
            y_train = st.session_state["y_train"]
            y_test = st.session_state["y_test"]

            with st.spinner(f"Training Neural Network ({nn_epochs} epochs)…"):
                nn = build_neural_network(X_train.shape[1])
                nn = train_neural_network(nn, X_train, y_train, epochs=nn_epochs)

            # Save trained model
            os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
            nn.save(NN_MODEL_PATH)
            st.session_state["nn_model"] = nn
            st.success("✅ Neural Network trained & saved!")

            # Evaluate
            y_prob = predict_neural_network(nn, X_test).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            metrics = evaluate_model(y_test, y_pred, y_prob)
            st.session_state["nn_metrics"] = metrics
            st.session_state["nn_y_pred"] = y_pred
            st.session_state["nn_y_prob"] = y_prob

            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
            st.metric("F1", f"{metrics['f1']:.4f}")
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

        # Auto-evaluate if model loaded but no metrics yet
        if (st.session_state["nn_model"] is not None
                and st.session_state["X_test"] is not None
                and st.session_state["nn_metrics"] is None
                and not train_nn):
            nn = st.session_state["nn_model"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            y_prob = predict_neural_network(nn, X_test).flatten()
            y_pred = (y_prob > 0.5).astype(int)
            metrics = evaluate_model(y_test, y_pred, y_prob)
            st.session_state["nn_metrics"] = metrics
            st.session_state["nn_y_pred"] = y_pred
            st.session_state["nn_y_prob"] = y_prob
            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

    # ── K-Means ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📍 K-Means Clustering (Unsupervised)")
    if st.button("Run K-Means"):
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        with st.spinner("Running K-Means…"):
            _, clusters = train_kmeans(X_test, n_clusters=2)
            ct = evaluate_kmeans(y_test, clusters)
        st.write("**Cluster vs Actual Label:**")
        st.dataframe(ct)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results":
    st.title("📈 Results — Model Comparison")

    rf_metrics = st.session_state["rf_metrics"]
    nn_metrics = st.session_state["nn_metrics"]

    if rf_metrics is None and nn_metrics is None:
        st.warning("Train or evaluate at least one model on the **Train Models** page first.")
        st.stop()

    # ── Summary Table ────────────────────────────────────────────────────────
    st.subheader("📋 Metrics Summary")
    rows = []
    if rf_metrics is not None:
        row = {"Model": "Random Forest"}
        for k in ["recall", "precision", "f1", "accuracy"]:
            row[k.capitalize()] = round(rf_metrics[k], 4)
        if "auc_roc" in rf_metrics:
            row["AUC-ROC"] = round(rf_metrics["auc_roc"], 4)
        rows.append(row)
    if nn_metrics is not None:
        row = {"Model": "Neural Network"}
        for k in ["recall", "precision", "f1", "accuracy"]:
            row[k.capitalize()] = round(nn_metrics[k], 4)
        if "auc_roc" in nn_metrics:
            row["AUC-ROC"] = round(nn_metrics["auc_roc"], 4)
        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.table(metrics_df)

    # ── Metrics Bar Chart ────────────────────────────────────────────────────
    st.subheader("📊 Metrics Comparison")
    chart_cols = [c for c in metrics_df.columns if c != "Accuracy"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(chart_cols))
    width = 0.35
    if rf_metrics is not None:
        vals = [metrics_df.loc["Random Forest", c] for c in chart_cols]
        ax.bar(x - width/2, vals, width, label="Random Forest", color="#2196F3")
    if nn_metrics is not None:
        vals = [metrics_df.loc["Neural Network", c] for c in chart_cols]
        ax.bar(x + width/2, vals, width, label="Neural Network", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(chart_cols)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title("Model Comparison")
    st.pyplot(fig)

    # ── Confusion Matrices ───────────────────────────────────────────────────
    st.subheader("🔢 Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    import seaborn as sns

    if rf_metrics is not None:
        with cm_col1:
            st.caption("Random Forest")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                rf_metrics["confusion_matrix"], annot=True, fmt="d",
                cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    if nn_metrics is not None:
        with cm_col2:
            st.caption("Neural Network")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                nn_metrics["confusion_matrix"], annot=True, fmt="d",
                cmap="Oranges", ax=ax,
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    # ── ROC Curve ────────────────────────────────────────────────────────────
    y_test = st.session_state["y_test"]
    rf_y_prob = st.session_state["rf_y_prob"]
    nn_y_prob = st.session_state["nn_y_prob"]

    if rf_y_prob is not None or nn_y_prob is not None:
        st.subheader("📉 ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        if rf_y_prob is not None:
            fpr, tpr, _ = get_roc_curve(y_test, rf_y_prob)
            ax.plot(fpr, tpr, label=f"Random Forest (AUC={rf_metrics['auc_roc']:.4f})", linewidth=2, color="#2196F3")
        if nn_y_prob is not None:
            fpr, tpr, _ = get_roc_curve(y_test, nn_y_prob)
            ax.plot(fpr, tpr, label=f"Neural Network (AUC={nn_metrics['auc_roc']:.4f})", linewidth=2, color="#FF9800")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # ── Precision-Recall Curve ───────────────────────────────────────────────
    if rf_y_prob is not None or nn_y_prob is not None:
        st.subheader("📉 Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        if rf_y_prob is not None:
            prec, rec, _ = get_pr_curve(y_test, rf_y_prob)
            ap = rf_metrics.get("avg_precision", 0)
            ax.plot(rec, prec, label=f"Random Forest (AP={ap:.4f})", linewidth=2, color="#2196F3")
        if nn_y_prob is not None:
            prec, rec, _ = get_pr_curve(y_test, nn_y_prob)
            ap = nn_metrics.get("avg_precision", 0)
            ax.plot(rec, prec, label=f"Neural Network (AP={ap:.4f})", linewidth=2, color="#FF9800")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # ── Feature Importance ───────────────────────────────────────────────────
    if st.session_state["rf_model"] is not None and st.session_state["feature_names"] is not None:
        st.markdown("---")
        st.subheader("🌲 Feature Importance (Random Forest)")
        importances = get_feature_importances(st.session_state["rf_model"])
        feature_names = st.session_state["feature_names"]
        fig_feat = plot_feature_importance(importances, feature_names)
        st.pyplot(fig_feat)
        plt.close(fig_feat)

    # ── Accuracy caveat ──────────────────────────────────────────────────────
    st.info(
        "⚠️ **Note on Accuracy:** This dataset is highly imbalanced (~0.17% fraud). "
        "A model predicting *all legit* would achieve 99.83% accuracy. "
        "Focus on **Recall**, **Precision**, **F1**, and **AUC-ROC** for meaningful evaluation."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Single Transaction Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Single Transaction":
    st.title("🔍 Single Transaction Prediction")

    if st.session_state["rf_model"] is None and st.session_state["nn_model"] is None:
        st.warning("Load or train at least one model first.")
        st.stop()

    feature_names = st.session_state["feature_names"]
    scaler = st.session_state["scaler"]

    if feature_names is None:
        st.warning("Feature names not available. Please load data first.")
        st.stop()

    # ── Mode selector ────────────────────────────────────────────────────────
    mode = st.radio("Prediction Mode", ["Single Entry", "Batch CSV Upload"], horizontal=True)

    if mode == "Single Entry":
        if scaler is None:
            st.warning(
                "⚠️ No scaler found. Predictions may be inaccurate because "
                "models expect scaled inputs. Preprocess data on the Train page first."
            )

        st.markdown("Enter feature values for a transaction:")

        with st.form("predict_form"):
            cols = st.columns(5)
            inputs = {}
            for i, feat in enumerate(feature_names):
                inputs[feat] = cols[i % 5].number_input(feat, value=0.0, format="%.4f")
            submitted = st.form_submit_button("Predict", use_container_width=True)

        if submitted:
            row = np.array([[inputs[f] for f in feature_names]])

            # Scale if scaler is available
            if scaler is not None:
                row = scaler.transform(row)

            col1, col2 = st.columns(2)

            if st.session_state["rf_model"] is not None:
                rf = st.session_state["rf_model"]
                pred = rf.predict(row)[0]
                prob = rf.predict_proba(row)[0][1]
                with col1:
                    st.subheader("🌲 Random Forest")
                    if pred == 1:
                        st.error(f"🚨 FRAUD DETECTED  (probability: {prob:.4f})")
                    else:
                        st.success(f"✅ Legitimate  (fraud probability: {prob:.4f})")
                    st.progress(float(prob))

            if st.session_state["nn_model"] is not None:
                nn = st.session_state["nn_model"]
                raw = predict_neural_network(nn, row)
                prob_nn = float(np.squeeze(raw))
                pred_nn = int(prob_nn > 0.5)
                with col2:
                    st.subheader("🧠 Neural Network")
                    if pred_nn == 1:
                        st.error(f"🚨 FRAUD DETECTED  (probability: {prob_nn:.4f})")
                    else:
                        st.success(f"✅ Legitimate  (fraud probability: {prob_nn:.4f})")
                    st.progress(prob_nn)

    # ── Batch CSV Upload ─────────────────────────────────────────────────────
    else:
        st.markdown("Upload a CSV with the same feature columns (without `Class`) for batch predictions.")
        batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")

        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("Run Batch Prediction", use_container_width=True):
                # Validate columns
                missing_cols = set(feature_names) - set(batch_df.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.stop()

                X_batch = batch_df[feature_names].values

                # Scale if scaler is available
                if scaler is not None:
                    X_batch = scaler.transform(X_batch)
                else:
                    st.warning("No scaler — predictions may be inaccurate.")

                results = batch_df.copy()

                if st.session_state["rf_model"] is not None:
                    rf = st.session_state["rf_model"]
                    results["RF_Prediction"] = rf.predict(X_batch)
                    results["RF_Fraud_Prob"] = rf.predict_proba(X_batch)[:, 1]

                if st.session_state["nn_model"] is not None:
                    nn = st.session_state["nn_model"]
                    probs = predict_neural_network(nn, X_batch).flatten()
                    results["NN_Prediction"] = (probs > 0.5).astype(int)
                    results["NN_Fraud_Prob"] = probs

                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True)

                # Summary
                if "RF_Prediction" in results.columns:
                    rf_fraud = int(results["RF_Prediction"].sum())
                    st.info(f"🌲 RF flagged **{rf_fraud}** / {len(results)} transactions as fraud.")
                if "NN_Prediction" in results.columns:
                    nn_fraud = int(results["NN_Prediction"].sum())
                    st.info(f"🧠 NN flagged **{nn_fraud}** / {len(results)} transactions as fraud.")

                # Download button
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )