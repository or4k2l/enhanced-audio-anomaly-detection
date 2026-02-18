import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Synthetic audio generation function
def generate_synthetic_audio(duration=2, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    # Create a synthetic audio signal (e.g., sine wave)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.5 * np.random.normal(size=t.shape)
    return audio

# Feature extraction function

def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Generate synthetic audio
synthetic_audio = generate_synthetic_audio()
features = extract_features(synthetic_audio)

# Prepare dataset
X = []  # Feature set
Y = []  # Labels
for _ in range(100):  # Generate 100 samples
    audio = generate_synthetic_audio()
    feature = extract_features(audio)
    X.append(feature)
    Y.append(1)  # Label for synthetic audio

# Convert to DataFrame for easier handling
X = pd.DataFrame(X)
Y = pd.Series(Y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)

# Train Autoencoder model
autoencoder = Sequential()
autoencoder.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dense(X_train_scaled.shape[1], activation='sigmoid'))
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=10, validation_data=(X_test_scaled, X_test_scaled))

# Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_predictions))
