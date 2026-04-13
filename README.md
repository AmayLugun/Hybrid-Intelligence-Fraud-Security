# Hybrid Intelligence Fraud Security 🛡️💹

An advanced financial fraud detection system leveraging **Hybrid Intelligence** (Supervised + Unsupervised + Deep Learning) and high-fidelity visualizations to secure financial transactions.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Overview
This project provides a robust framework for detecting fraudulent financial activities in highly imbalanced datasets. It combines **Random Forest**, **Neural Networks**, and **K-Means Clustering** to create a multi-layered defense mechanism. The system is designed to identify subtle fraud patterns that traditional rule-based systems often miss.

## 🚀 Key Features
- **Data Preprocessing**: Automated handling of imbalanced datasets using **SMOTE** (Synthetic Minority Over-sampling Technique) and robust **StandardScaling**.
- **Hybrid Modeling Engine**:
    - `Random Forest`: High-precision classification and feature importance ranking.
    - `Neural Network`: Deep learning architecture for capturing non-linear fraud patterns.
    - `K-Means`: Unsupervised anomaly detection for discovering emerging fraud clusters.
- **Interactive Dashboard**: A streamlined Streamlit app featuring 4 specialized modules:
    - **📊 Data Explorer**: Statistical breakdowns, correlation insights, and feature distribution analysis.
    - **🤖 Training Hub**: On-demand model training and auto-loading of pre-trained weights.
    - **📈 Results Lab**: Comparative performance metrics (Recall, Precision, AUC-ROC).
    - **🔍 Single Prediction**: Real-time fraud scoring for individual transactions.
- **Advanced Visualizations**: Optimized horizontal correlation bars and class-separation density plots to make data actionable.

## 📂 Project Structure
```text
├── app.py                  # Main Streamlit Dashboard
├── requirements.txt        # Project Dependencies
├── src/                    # Core Analytical Logic
│   ├── preprocess.py       # SMOTE & Data Scaling
│   ├── evaluation.py       # Metrics & Curve Generators
│   ├── visualization.py    # Custom Plotting Library (Heatmaps, Distributions)
│   ├── random_forest_model.py
│   ├── neural_network.py
│   └── kmeans.py
├── models/                 # Serialized Models (.pkl, .keras)
├── notebook/               # Research & EDA Notebooks
└── data/                   # Dataset Directory (creditcard.csv)
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AmayLugun/Hybrid-Intelligence-Fraud-Security-.git
   cd Hybrid-Intelligence-Fraud-Security-
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset
The project utilizes the **Credit Card Fraud Detection** dataset (September 2013 European cardholders).
- **Total Transactions**: 284,807
- **Fraud Rate**: 0.17% (Extremely imbalanced)
- **Features**: Time, Amount, and 28 PCA-transformed variables (V1-V28).

> [!TIP]
> The included notebook handles data fetching automatically via `kagglehub`.

## 💻 Usage
To launch the interactive dashboard:
```bash
streamlit run app.py
```

---
**Developed by [Amay Lugun](https://github.com/AmayLugun)**  
*Securing the future of digital transactions through Hybrid Intelligence.*
