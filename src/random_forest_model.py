from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix

def train_random_forest(X, y, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model

def get_feature_importances(model):
    return model.feature_importances_