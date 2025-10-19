import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------------
# Configuration
# ------------------------------
dataset_path = r"D:\My Files\Proj\Python\music_dataset"
classes = ['happy', 'sad', 'tender']
model_file = r"D:\My Files\Proj\Python\model.pkl"

# ------------------------------
# Feature extraction
# ------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        # Pad if too short
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)), mode='constant')

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

        features = np.concatenate((mfccs, chroma, contrast, tonnetz))
        return features.reshape(1, -1)  # Always 2D
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# ------------------------------
# Model Training
# ------------------------------
def train_model():
    print("Starting training...")
    X, y_labels = [], []

    for label in classes:
        folder = os.path.join(dataset_path, label)
        if not os.path.isdir(folder):
            print(f"Warning: Directory not found for class '{label}'")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(('.mp3', '.wav', '.ogg'))]
        if not files:
            print(f"Warning: No audio files in '{folder}'")
            continue

        for file in files:
            path = os.path.join(folder, file)
            features = extract_features(path)
            if features is not None:
                X.append(features.flatten())
                y_labels.append(label)

    if not X:
        raise ValueError("No audio files found. Check dataset_path and subfolders!")

    X = np.array(X)
    y_labels = np.array(y_labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(clf, model_file)
    print(f"Model saved as {model_file}")

if __name__ == "__main__":
    train_model()
