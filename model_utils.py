import numba_fallback
import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display
import io
import base64

# ---- Class info ----
CLASS_EMOJIS = {
    "air_conditioner": "❄️",
    "car_horn": "📯",
    "children_playing": "👧",
    "dog_bark": "🐕",
    "drilling": "🔧",
    "engine_idling": "🚗",
    "gun_shot": "💥",
    "jackhammer": "🔨",
    "siren": "🚨",
    "street_music": "🎵"
}

CLASS_DESCRIPTIONS = {
    "air_conditioner": "A continuous mechanical hum from HVAC systems",
    "car_horn": "Short sharp beeping from vehicle horns",
    "children_playing": "Irregular voices and activity sounds from kids",
    "dog_bark": "Sharp impulsive sounds from barking dogs",
    "drilling": "Repetitive mechanical drilling noise",
    "engine_idling": "Low continuous rumble of a running engine",
    "gun_shot": "Single sharp high-energy impulse sound",
    "jackhammer": "Repetitive heavy impact drilling sound",
    "siren": "Rising and falling wail from emergency vehicles",
    "street_music": "Mixed musical sounds from street performers"
}

# Global variables
svm_model = None
lstm_model = None
scaler = None
label_encoder = None
class_names = None
LSTM_MFCC_FRAMES = 174

def load_or_train_model():
    """Load saved model if exists, otherwise train from scratch"""
    global svm_model, lstm_model, scaler, label_encoder, class_names

    model_path = "models/svm_model.pkl"
    lstm_model_path = "models/lstm_model.keras"
    scaler_path = "models/scaler.pkl"
    encoder_path = "models/label_encoder.pkl"

    if os.path.exists(model_path):
        print("Loading saved model...")
        svm_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        class_names = label_encoder.classes_
        print("Model loaded!")
    else:
        print("Training model for first time... (takes 2-3 min)")
        df = pd.read_csv("features_mfcc.csv")
        X = df.drop(columns=["label", "fold"]).values
        y = df["label"].values

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        svm_model = SVC(kernel="rbf", probability=True, random_state=42)
        svm_model.fit(X_scaled, y_encoded)

        # Save for next time
        joblib.dump(svm_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoder, encoder_path)
        print("Model trained and saved!")

    if os.path.exists(lstm_model_path):
        try:
            from tensorflow import keras

            print("Loading saved LSTM model...")
            lstm_model = keras.models.load_model(lstm_model_path)
            print("LSTM model loaded!")
        except Exception as e:
            print(f"Notice: Could not load LSTM model ({e}). SVM model is active.")
            lstm_model = None

def _extract_lstm_features(y, sr):
    """Return a fixed-length MFCC sequence for the LSTM model."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    if mfccs.shape[0] < LSTM_MFCC_FRAMES:
        mfccs = np.pad(
            mfccs,
            ((0, LSTM_MFCC_FRAMES - mfccs.shape[0]), (0, 0)),
            mode="constant",
        )
    else:
        mfccs = mfccs[:LSTM_MFCC_FRAMES]

    # Normalize each clip consistently with the training script.
    mfccs = (mfccs - mfccs.mean()) / (mfccs.std() + 1e-8)
    return mfccs[np.newaxis, ...]


def predict_audio(file_path, model_name="svm"):
    """Extract features from audio file and predict class"""
    # Load audio
    y, sr = librosa.load(file_path, duration=4.0)

    if model_name == "lstm":
        if lstm_model is None:
            raise RuntimeError(
                "The LSTM model is currently unavailable (requires trained models/lstm_model.keras and TensorFlow). "
                "Please select 'SVM (RBF Kernel)' to classify your audio."
            )
        proba = lstm_model.predict(_extract_lstm_features(y, sr), verbose=0)[0]
        model_label = "LSTM"
    elif model_name == "svm":
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1).reshape(1, -1)
        mfccs_scaled = scaler.transform(mfccs_mean)
        proba = svm_model.predict_proba(mfccs_scaled)[0]
        model_label = "SVM (RBF Kernel)"
    else:
        raise ValueError("Unsupported model. Choose 'svm' or 'lstm'.")

    pred_idx = np.argmax(proba)
    pred_class = class_names[pred_idx]

    # Top 3
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {
            "class": class_names[i].replace("_", " ").title(),
            "raw": class_names[i],
            "emoji": CLASS_EMOJIS.get(class_names[i], "🔊"),
            "confidence": round(proba[i] * 100, 1)
        }
        for i in top3_idx
    ]

    # Generate visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), facecolor="#1e293b")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#6366f1")
    ax1.set_title("Waveform", color="#e2e8f0", fontsize=10)
    ax1.set_xlabel("")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                             y_axis="mel", ax=ax2, cmap="magma")
    ax2.set_title("Mel Spectrogram", color="#e2e8f0", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100,
                bbox_inches="tight", facecolor="#1e293b")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return {
        "prediction": pred_class.replace("_", " ").title(),
        "raw": pred_class,
        "emoji": CLASS_EMOJIS.get(pred_class, "🔊"),
        "description": CLASS_DESCRIPTIONS.get(pred_class, ""),
        "confidence": round(proba[pred_idx] * 100, 1),
        "top3": top3,
        "waveform": img_b64,
        "duration": round(len(y) / sr, 2),
        "sample_rate": sr,
        "model": model_label
    }