from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import recall_score, confusion_matrix

def build_neural_network(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])
    return model

def train_neural_network(model, X, y, epochs=10, batch_size=32, validation_split=0.2):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return model

def predict_neural_network(model, X):
    return model.predict(X)

def evaluate_neural_network(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return recall, cm