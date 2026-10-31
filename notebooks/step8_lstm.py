"""Train an LSTM on time-ordered MFCC sequences from UrbanSound8K.

Run from the project root after downloading UrbanSound8K:
    python notebooks/step8_lstm.py
"""

import os

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


DATASET_DIR = "UrbanSound8K"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata", "UrbanSound8K.csv")
MODEL_PATH = "models/lstm_model.keras"
ENCODER_PATH = "models/lstm_label_encoder.pkl"
SAMPLE_RATE = 22050
CLIP_DURATION = 4.0
N_MFCC = 40
MFCC_FRAMES = 174


def extract_sequence(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=CLIP_DURATION)
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC).T
    if mfccs.shape[0] < MFCC_FRAMES:
        mfccs = np.pad(
            mfccs,
            ((0, MFCC_FRAMES - mfccs.shape[0]), (0, 0)),
            mode="constant",
        )
    else:
        mfccs = mfccs[:MFCC_FRAMES]
    return (mfccs - mfccs.mean()) / (mfccs.std() + 1e-8)


def load_dataset():
    metadata = pd.read_csv(METADATA_PATH)
    sequences, labels = [], []

    for index, row in metadata.iterrows():
        file_path = os.path.join(
            DATASET_DIR,
            "audio",
            "fold" + str(row["fold"]),
            row["slice_file_name"],
        )
        try:
            sequences.append(extract_sequence(file_path))
            labels.append(row["class"])
        except Exception as error:
            print(f"Skipping {file_path}: {error}")
        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1} / {len(metadata)} files...")

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels)


def build_model(num_classes):
    return keras.Sequential(
        [
            keras.layers.Input(shape=(MFCC_FRAMES, N_MFCC)),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def main():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"{METADATA_PATH} was not found. Download and extract UrbanSound8K first."
        )

    print("Extracting MFCC sequences...")
    X, labels = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(len(encoder.classes_))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"LSTM test accuracy: {test_accuracy * 100:.1f}%")
    print(
        classification_report(
            y_test, predictions, target_names=encoder.classes_, zero_division=0
        )
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    # Keep the encoder alongside the model for reproducible deployments.
    import joblib

    joblib.dump(encoder, ENCODER_PATH)
    print(f"Saved {MODEL_PATH} and {ENCODER_PATH}")


if __name__ == "__main__":
    main()
