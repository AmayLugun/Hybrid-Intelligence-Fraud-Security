import pandas as pd
from sklearn.cluster import KMeans

def train_kmeans(X, n_clusters=2, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster = model.fit_predict(X)
    return model, cluster

def evaluate_kmeans(y, cluster):
    df = pd.DataFrame({'Actual': y, 'Cluster': cluster})
    return pd.crosstab(df['Actual'], df['Cluster'])