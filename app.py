from flask import Flask, render_template, request, jsonify
import os
from model_utils import load_or_train_model, predict_audio, CLASS_EMOJIS, CLASS_DESCRIPTIONS

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp"
os.makedirs("temp", exist_ok=True)

# Load model when app starts
load_or_train_model()

# ---- Dashboard data ----
DASHBOARD_DATA = {
    "total_files": 8732,
    "num_classes": 10,
    "best_accuracy": 87.0,
    "models": {
        "names": ["Logistic Regression", "SVM", "Random Forest"],
        "accuracies": [42.7, 87.0, 50.6],
        "colors": ["#4f46e5", "#7c3aed", "#6d28d9"]
    },
    "classes": [
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer",
        "siren", "street_music"
    ],
    "f1_scores": [89, 92, 76, 82, 89, 93, 83, 93, 94, 76],
    "confused_pairs": [
        {"actual": "street_music", "predicted": "children_playing",
         "count": 102, "reason": "Both have voices and irregular outdoor sounds"},
        {"actual": "children_playing", "predicted": "street_music",
         "count": 71, "reason": "Overlapping frequency and energy patterns"},
        {"actual": "dog_bark", "predicted": "children_playing",
         "count": 70, "reason": "Both are sharp, irregular impulse sounds"},
        {"actual": "street_music", "predicted": "air_conditioner",
         "count": 53, "reason": "Both can be continuous droning sounds"},
        {"actual": "gun_shot", "predicted": "dog_bark",
         "count": 50, "reason": "Both are short, high-energy impulse sounds"},
    ]
}

@app.route("/")
def home():
    return render_template("index.html", emojis=CLASS_EMOJIS)

@app.route("/classifier")
def classifier():
    return render_template("classifier.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", data=DASHBOARD_DATA,
                           emojis=CLASS_EMOJIS)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    temp_path = os.path.join("temp", "upload.wav")
    file.save(temp_path)

    try:
        model_name = request.form.get("model", "svm").lower()
        result = predict_audio(temp_path, model_name=model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)