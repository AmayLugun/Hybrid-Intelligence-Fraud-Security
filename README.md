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
This project uses the **Credit Card Fraud Detection** dataset (highly imbalanced).
- **Format**: CSV
- **Download**: You can download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Placement**: Place the `creditcard.csv` file inside the `data/` folder before running the application.

## 💻 Usage
To launch the interactive dashboard:
```bash
streamlit run app.py
```

---
Created by [Amay Lugun](https://github.com/AmayLugun)
