import os
import io
import time
import base64
import zipfile
import h5py
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    "air_conditioner": "Continuous mechanical hum from HVAC systems",
    "car_horn": "Short sharp acoustic pulse from vehicle warning horns",
    "children_playing": "Irregular voices and multi-speaker activity sounds",
    "dog_bark": "Sharp impulsive acoustic transients from canine vocalizations",
    "drilling": "High-frequency repetitive mechanical friction noise",
    "engine_idling": "Low-frequency stationary rumble of internal combustion engines",
    "gun_shot": "High-energy instantaneous blast wave and decay envelope",
    "jackhammer": "Low-frequency repetitive heavy impact percussion sounds",
    "siren": "Harmonically rich frequency-modulated emergency vehicle signal",
    "street_music": "Polyrhythmic acoustic mixture of melodies and percussions"
}

# ---- PyTorch 2D CNN Architecture ----
class PyTorchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(25088, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        return self.softmax(self.fc2(x))

# ---- PyTorch LSTM Architecture (Matches Keras sequential LSTM) ----
class PyTorchLSTM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x: (batch, 40, 1)
        out, _ = self.lstm1(x)
        out, (hn, cn) = self.lstm2(out)
        last_out = hn.squeeze(0)
        x = self.relu(self.fc1(last_out))
        return self.softmax(self.fc2(x))

# Global registry
svm_model = None
rf_model = None
lr_model = None
cnn_model = None
lstm_model = None
scaler = None
label_encoder = None
class_names = None

MODEL_DISPLAY_NAMES = {
    "svm": "Support Vector Machine (RBF)",
    "rf": "Random Forest (100 Trees)",
    "lr": "Logistic Regression",
    "cnn": "2D Mel-Spectrogram CNN",
    "lstm": "Recurrent Neural Network (LSTM)",
    "ensemble": "Hybrid Late-Fusion Ensemble (CNN + SVM + RF + LSTM)",
    "all": "Model Arena (Benchmark All Models)"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_or_train_model():
    """Load all models (SVM, RF, LR, CNN, LSTM) into memory."""
    global svm_model, rf_model, lr_model, cnn_model, lstm_model, scaler, label_encoder, class_names
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    lr_path = os.path.join(MODELS_DIR, "lr_model.pkl")
    cnn_pt_path = os.path.join(MODELS_DIR, "cnn_model.pt")
    lstm_keras_path = os.path.join(MODELS_DIR, "lstm_model.keras")

    print("Loading model registry...")
    if os.path.exists(encoder_path) and os.path.exists(scaler_path):
        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        class_names = list(label_encoder.classes_)
    else:
        raise FileNotFoundError("Preprocessor artifacts missing from models/")

    if os.path.exists(svm_path):
        svm_model = joblib.load(svm_path)
        print("  [OK] SVM loaded")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        print("  [OK] Random Forest loaded")
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)
        print("  [OK] Logistic Regression loaded")
        
    if os.path.exists(cnn_pt_path):
        cnn_model = PyTorchCNN(num_classes=len(class_names))
        cnn_model.load_state_dict(torch.load(cnn_pt_path, map_location=torch.device('cpu')))
        cnn_model.eval()
        print("  [OK] 2D CNN (PyTorch) loaded")

    if os.path.exists(lstm_keras_path):
        try:
            z = zipfile.ZipFile(lstm_keras_path)
            with z.open('model.weights.h5') as f:
                bio = io.BytesIO(f.read())
            h = h5py.File(bio, 'r')
            lstm_model = PyTorchLSTM(num_classes=len(class_names))
            with torch.no_grad():
                lstm_model.lstm1.weight_ih_l0.copy_(torch.from_numpy(h['layers/lstm/cell/vars/0'][:].T))
                lstm_model.lstm1.weight_hh_l0.copy_(torch.from_numpy(h['layers/lstm/cell/vars/1'][:].T))
                lstm_model.lstm1.bias_ih_l0.copy_(torch.from_numpy(h['layers/lstm/cell/vars/2'][:]))
                lstm_model.lstm1.bias_hh_l0.zero_()
                
                lstm_model.lstm2.weight_ih_l0.copy_(torch.from_numpy(h['layers/lstm_1/cell/vars/0'][:].T))
                lstm_model.lstm2.weight_hh_l0.copy_(torch.from_numpy(h['layers/lstm_1/cell/vars/1'][:].T))
                lstm_model.lstm2.bias_ih_l0.copy_(torch.from_numpy(h['layers/lstm_1/cell/vars/2'][:]))
                lstm_model.lstm2.bias_hh_l0.zero_()
                
                lstm_model.fc1.weight.copy_(torch.from_numpy(h['layers/dense/vars/0'][:].T))
                lstm_model.fc1.bias.copy_(torch.from_numpy(h['layers/dense/vars/1'][:]))
                
                lstm_model.fc2.weight.copy_(torch.from_numpy(h['layers/dense_1/vars/0'][:].T))
                lstm_model.fc2.bias.copy_(torch.from_numpy(h['layers/dense_1/vars/1'][:]))
            lstm_model.eval()
            print("  [OK] LSTM (Recurrent Neural Network) loaded from keras artifact")
        except Exception as e:
            print(f"  [!] Note: LSTM could not be initialized from artifact: {e}")

    print(f"Model registry ready with {len(class_names)} noise categories.")

def inject_noise(y, snr_db):
    """Simulate additive acoustic background noise at a specific SNR (dB)."""
    if snr_db is None or str(snr_db).lower() in ["clean", "none", ""]:
        return y, None
    try:
        snr = float(snr_db)
    except ValueError:
        return y, None
        
    signal_power = np.mean(y ** 2)
    if signal_power <= 1e-12:
        return y, snr
        
    noise_power = signal_power / (10.0 ** (snr / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise, snr

def extract_features(y, sr):
    """Extract both 1D statistical MFCCs and 2D Mel-Spectrogram."""
    # 1. 1D MFCCs for Traditional Models
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1).reshape(1, -1)
    mfccs_scaled = scaler.transform(mfccs_mean)
    
    # 2. 2D Normalized Mel-Spectrogram for CNN
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    t_mel = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).float()
    t_resized = F.interpolate(t_mel, size=(128, 128), mode='bilinear', align_corners=False)
    t_norm = (t_resized - t_resized.min()) / (t_resized.max() - t_resized.min() + 1e-8)
    
    # 3. Acoustic Signal Telemetry
    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    s_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    
    telemetry = {
        "rms": round(rms, 4),
        "zcr": round(zcr, 4),
        "spectral_centroid": round(sc, 1),
        "spectral_rolloff": round(s_rolloff, 1),
        "mfccs_vector": [round(float(v), 2) for v in mfccs_mean[0][:15]]  # First 15 for visualization
    }
    
    return mfccs_scaled, t_norm, telemetry

def _predict_single_model(model_key, mfccs_scaled, t_mel_tensor):
    """Predict probabilities and measure latency for a single model."""
    start_t = time.perf_counter()
    
    if model_key == "svm":
        proba = svm_model.predict_proba(mfccs_scaled)[0]
    elif model_key == "rf":
        proba = rf_model.predict_proba(mfccs_scaled)[0]
    elif model_key == "lr":
        proba = lr_model.predict_proba(mfccs_scaled)[0]
    elif model_key == "cnn":
        with torch.no_grad():
            proba = cnn_model(t_mel_tensor)[0].numpy()
    elif model_key == "lstm":
        if lstm_model is not None:
            t_seq = torch.from_numpy(mfccs_scaled).unsqueeze(-1).float()
            with torch.no_grad():
                proba = lstm_model(t_seq)[0].numpy()
        else:
            proba = svm_model.predict_proba(mfccs_scaled)[0]
    elif model_key == "ensemble":
        p_svm = svm_model.predict_proba(mfccs_scaled)[0]
        p_rf = rf_model.predict_proba(mfccs_scaled)[0]
        with torch.no_grad():
            p_cnn = cnn_model(t_mel_tensor)[0].numpy()
        if lstm_model is not None:
            t_seq = torch.from_numpy(mfccs_scaled).unsqueeze(-1).float()
            with torch.no_grad():
                p_lstm = lstm_model(t_seq)[0].numpy()
            proba = (p_svm * 0.25 + p_rf * 0.25 + p_cnn * 0.25 + p_lstm * 0.25)
        else:
            proba = (p_svm * 0.35 + p_rf * 0.35 + p_cnn * 0.30)
    else:
        raise ValueError(f"Unknown model: {model_key}")
        
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    return proba, latency_ms

def predict_audio(file_path, model_name="svm", snr=None):
    """Main inference entrypoint with SNR robustness & multi-model support."""
    y, sr = librosa.load(file_path, duration=4.0)
    
    # Inject synthetic noise if requested
    y, simulated_snr = inject_noise(y, snr)
    
    # Extract features
    mfccs_scaled, t_mel, telemetry = extract_features(y, sr)
    
    # Generate visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), facecolor="#1e293b")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#6366f1")
    title_suffix = f" (Simulated SNR: {simulated_snr} dB)" if simulated_snr is not None else ""
    ax1.set_title(f"Acoustic Waveform{title_suffix}", color="#e2e8f0", fontsize=10)
    ax1.set_xlabel("")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
    ax2.set_title("Mel Spectrogram", color="#e2e8f0", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#1e293b")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    
    model_name = model_name.lower()
    
    # Model Arena Mode: run all models
    if model_name == "all":
        arena_results = []
        model_keys = ["svm", "rf", "lr", "cnn"]
        if lstm_model is not None:
            model_keys.append("lstm")
        model_keys.append("ensemble")
        for key in model_keys:
            proba, lat = _predict_single_model(key, mfccs_scaled, t_mel)
            pred_idx = np.argmax(proba)
            pred_c = class_names[pred_idx]
            arena_results.append({
                "key": key,
                "name": MODEL_DISPLAY_NAMES[key],
                "prediction": pred_c.replace("_", " ").title(),
                "raw_class": pred_c,
                "emoji": CLASS_EMOJIS.get(pred_c, "🔊"),
                "confidence": round(float(proba[pred_idx]) * 100, 1),
                "latency_ms": lat
            })
            
        # Consensus analysis
        preds = [r["raw_class"] for r in arena_results]
        unique_p, counts = np.unique(preds, return_counts=True)
        majority_idx = np.argmax(counts)
        consensus_class = unique_p[majority_idx]
        consensus_agreement = f"{counts[majority_idx]} of {len(arena_results)} models agree"
        
        # Primary result defaults to ensemble
        primary_proba, p_lat = _predict_single_model("ensemble", mfccs_scaled, t_mel)
        pred_idx = np.argmax(primary_proba)
        pred_class = class_names[pred_idx]
        
        top3_idx = np.argsort(primary_proba)[::-1][:3]
        top3 = [
            {
                "class": class_names[i].replace("_", " ").title(),
                "raw": class_names[i],
                "emoji": CLASS_EMOJIS.get(class_names[i], "🔊"),
                "confidence": round(float(primary_proba[i]) * 100, 1)
            }
            for i in top3_idx
        ]
        
        return {
            "prediction": pred_class.replace("_", " ").title(),
            "raw": pred_class,
            "emoji": CLASS_EMOJIS.get(pred_class, "🔊"),
            "description": CLASS_DESCRIPTIONS.get(pred_class, ""),
            "confidence": round(float(primary_proba[pred_idx]) * 100, 1),
            "top3": top3,
            "waveform": img_b64,
            "duration": round(len(y) / sr, 2),
            "sample_rate": sr,
            "model": "Model Arena (Multi-Model Comparison)",
            "telemetry": telemetry,
            "simulated_snr": simulated_snr,
            "arena": arena_results,
            "consensus": {
                "class": consensus_class.replace("_", " ").title(),
                "agreement": consensus_agreement
            }
        }
        
    # Single Model Mode
    target_key = model_name if model_name in MODEL_DISPLAY_NAMES else "svm"
    proba, latency_ms = _predict_single_model(target_key, mfccs_scaled, t_mel)
    
    pred_idx = np.argmax(proba)
    pred_class = class_names[pred_idx]
    
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {
            "class": class_names[i].replace("_", " ").title(),
            "raw": class_names[i],
            "emoji": CLASS_EMOJIS.get(class_names[i], "🔊"),
            "confidence": round(float(proba[i]) * 100, 1)
        }
        for i in top3_idx
    ]
    
    return {
        "prediction": pred_class.replace("_", " ").title(),
        "raw": pred_class,
        "emoji": CLASS_EMOJIS.get(pred_class, "🔊"),
        "description": CLASS_DESCRIPTIONS.get(pred_class, ""),
        "confidence": round(float(proba[pred_idx]) * 100, 1),
        "top3": top3,
        "waveform": img_b64,
        "duration": round(len(y) / sr, 2),
        "sample_rate": sr,
        "model": MODEL_DISPLAY_NAMES.get(target_key, target_key.upper()),
        "latency_ms": latency_ms,
        "telemetry": telemetry,
        "simulated_snr": simulated_snr,
        "arena": None
    }
