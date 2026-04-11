# Load dataset and Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df, test_size=0.2, use_smote=True):

    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split BEFORE scaling/oversampling to avoid information leakage from test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train
        
    return X_train_res, X_test_scaled, y_train_res, y_test, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
