# Noise Classification System

A Flask web application that classifies environmental audio into the ten
UrbanSound8K categories using a trained SVM model. The application extracts
MFCC features from an uploaded WAV/audio file and returns the predicted class,
confidence scores, and waveform/mel-spectrogram visualizations.

## Features

- SVM (RBF kernel) audio classification
- Optional LSTM prediction support when `models/lstm_model.keras` is present
- Top-three predictions with confidence scores
- Waveform and mel-spectrogram visualization
- Dashboard with model and class statistics
- Training notebooks and saved model artifacts

## Project structure

```text
app.py                  Flask application and routes
model_utils.py          Feature extraction, model loading, and prediction
models/                 Saved scaler, label encoder, and SVM model
notebooks/              Data exploration, feature extraction, and training scripts
templates/               Flask HTML templates
static/images/           Dashboard and analysis images
features_mfcc.csv       Extracted MFCC training features
urban_sound_cnn.keras   Saved CNN model artifact
requirements.txt        Python dependencies
```

## Setup

Python 3.10 through 3.12 is recommended for the web application. TensorFlow is
only needed when running the optional notebook training scripts; it is not
required for the Flask application.

```bash
git clone https://github.com/sam-rudddhi/Noise_Classification.git
cd Noise_Classification
git checkout feature/noise-classification
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

```bash
python app.py
```

Open http://127.0.0.1:5000 in a browser. Use the Classifier page to upload an
audio file and select SVM. The LSTM option requires a compatible
`models/lstm_model.keras` file and TensorFlow.

## Validation

Check all Python files for syntax errors:

```bash
python -m compileall .
```

The included `test.py` is a small environment sanity check and can be run
with `python test.py`.

The application loads the saved preprocessing and model artifacts from
`models/`. Keep this directory intact when cloning the repository.

## Notes

- Uploaded files are processed temporarily and removed after prediction.
- `urban_sound_cnn.keras` is larger than GitHub's recommended 50 MB file size.
  Git LFS is recommended if the model is updated or additional large artifacts
  are added.
- The project includes `numba_fallback.py` for environments where Numba's
  native dependencies cannot be loaded.