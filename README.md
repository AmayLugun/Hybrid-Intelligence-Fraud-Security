# Hybrid Intelligence Fraud Security 🛡️💹

An advanced financial fraud detection system leveraging hybrid intelligence (Machine Learning + Deep Learning) and real-time visualization to secure financial transactions.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)

## 🌟 Overview
This project provides a robust framework for detecting fraudulent financial activities. It combines multiple classification models (Random Forest, KMeans, Neural Networks) to provide a high-confidence prediction engine. The project includes a comprehensive Streamlit dashboard for data exploration and model performance monitoring.

## 🚀 Key Features
- **Data Preprocessing**: Advanced handling of imbalanced datasets using SMOTE and robust scaling.
- **Hybrid Modeling**: Utilizes both supervised (Random Forest, ANN) and unsupervised (K-Means) techniques.
- **Interactive Dashboard**: Real-time visualization of model metrics, feature importance, and transaction trends.
- **Performance Evaluation**: Detailed reports including Precision-Recall curves, AUC-ROC, and Confusion Matrices.

## 📂 Project Structure
```text
├── app.py              # Streamlit Web Application
├── src/                # Core Logic
│   ├── preprocess.py   # Data cleaning & Scaling
│   ├── evaluation.py   # Metric calculations
│   └── models/         # Model implementations (RF, KMeans, ANN)
├── notebook/           # Research and Development Notebooks
├── data/               # (Ignored) Raw transaction data
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AmayLugun/Hybrid-Intelligence-Fraud-Security-.git
   cd Hybrid-Intelligence-Fraud-Security-
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset
The project utilizes the **Credit Card Fraud Detection** dataset, which contains transactions made by credit cards in September 2013 by European cardholders.

### Key Characteristics:
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172% of the total) - **Highly Imbalanced**.
- **Features**: 
    - `Time`: Seconds elapsed between each transaction and the first transaction.
    - `Amount`: Transaction amount, which can be used for cost-sensitive learning.
    - `V1-V28`: Principal components obtained with PCA (for privacy reasons).
    - `Class`: Response variable (1 for fraud, 0 otherwise).

### Data Source:
You can find the dataset on [Kaggle](https://www.kaggle.com/datasets/jacklizhi/creditcard). 

> [!NOTE]
> The notebook included in this repository (`notebook/fraud_detection.ipynb`) uses `kagglehub` to automatically fetch the data, so manual download may be optional.

## 💻 Usage
To launch the interactive dashboard:
```bash
streamlit run app.py
```

---
Created by [Amay Lugun](https://github.com/AmayLugun)
