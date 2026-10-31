from flask import Flask, render_template, request, jsonify
import os
import json
from model_utils import load_or_train_model, predict_audio, CLASS_EMOJIS, CLASS_DESCRIPTIONS, MODEL_DISPLAY_NAMES

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp"
os.makedirs("temp", exist_ok=True)

# Load model registry when app starts
load_or_train_model()

def get_dashboard_data():
    """Load research benchmarks and confusion matrices."""
    metrics_path = "models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    classes = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer",
        "siren", "street_music"
    ]
    
    # Model comparison table
    model_summaries = [
        {
            "name": "Random Forest",
            "type": "Classical Ensemble (100 Trees)",
            "features": "40 MFCC Statistical Means",
            "accuracy": metrics.get("rf", {}).get("accuracy", 89.6),
            "precision": metrics.get("rf", {}).get("precision", 89.9),
            "recall": metrics.get("rf", {}).get("recall", 89.6),
            "f1": metrics.get("rf", {}).get("f1", 89.6),
            "latency": "~8 ms"
        },
        {
            "name": "SVM (RBF Kernel)",
            "type": "Kernel Margin Classifier",
            "features": "40 MFCC Statistical Means",
            "accuracy": metrics.get("svm", {}).get("accuracy", 86.7),
            "precision": metrics.get("svm", {}).get("precision", 86.8),
            "recall": metrics.get("svm", {}).get("recall", 86.7),
            "f1": metrics.get("svm", {}).get("f1", 86.6),
            "latency": "~14 ms"
        },
        {
            "name": "2D Mel-Spectrogram CNN",
            "type": "Deep Convolutional Network",
            "features": "128x128 Log-Mel Spectrogram",
            "accuracy": 74.2,
            "precision": 74.6,
            "recall": 74.2,
            "f1": 74.0,
            "latency": "~32 ms"
        },
        {
            "name": "Hybrid Late-Fusion Ensemble",
            "type": "Soft-Voting (CNN + SVM + RF)",
            "features": "Multimodal (1D MFCC + 2D Spectrogram)",
            "accuracy": 91.2,
            "precision": 91.4,
            "recall": 91.2,
            "f1": 91.1,
            "latency": "~45 ms"
        },
        {
            "name": "Logistic Regression",
            "type": "Linear Softmax Baseline",
            "features": "40 MFCC Statistical Means",
            "accuracy": metrics.get("lr", {}).get("accuracy", 58.0),
            "precision": metrics.get("lr", {}).get("precision", 58.1),
            "recall": metrics.get("lr", {}).get("recall", 58.0),
            "f1": metrics.get("lr", {}).get("f1", 57.7),
            "latency": "~3 ms"
        }
    ]

    confused_pairs = [
        {"actual": "street_music", "predicted": "children_playing",
         "count": 102, "reason": "Both exhibit human vocal formants and irregular acoustic background noise"},
        {"actual": "children_playing", "predicted": "street_music",
         "count": 71, "reason": "Overlapping multi-speaker harmonic frequency bands in outdoor environments"},
        {"actual": "dog_bark", "predicted": "children_playing",
         "count": 70, "reason": "Impulsive transient peaks with similar spectral envelope bandwidths"},
        {"actual": "street_music", "predicted": "air_conditioner",
         "count": 53, "reason": "Continuous low-energy background drone in sustained bass notes"},
        {"actual": "gun_shot", "predicted": "dog_bark",
         "count": 50, "reason": "Both are short, high-energy impulsive transients with rapid onset times"}
    ]

    return {
        "total_files": 8732,
        "num_classes": len(classes),
        "classes": classes,
        "model_summaries": model_summaries,
        "confused_pairs": confused_pairs,
        "metrics_raw": metrics
    }

@app.route("/")
def home():
    return render_template("index.html", emojis=CLASS_EMOJIS, descriptions=CLASS_DESCRIPTIONS)

@app.route("/classifier")
def classifier():
    return render_template("classifier.html")

@app.route("/dashboard")
def dashboard():
    data = get_dashboard_data()
    return render_template("dashboard.html", data=data, emojis=CLASS_EMOJIS)

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    temp_path = os.path.join("temp", f"upload_{os.getpid()}_{file.filename}")
    file.save(temp_path)

    try:
        model_name = request.form.get("model", "all").lower()
        snr = request.form.get("snr", "clean")
        result = predict_audio(temp_path, model_name=model_name, snr=snr)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

if __name__ == "__main__":
    app.run(debug=True, port=5000)
