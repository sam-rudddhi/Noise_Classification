import librosa
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# ---- Step 1: Extract Mel Spectrograms for all files ----
print("\nExtracting Mel Spectrograms...")
print("This will take 15-25 minutes. Please wait!\n")

df = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")

IMG_SIZE = 128   # we'll resize every spectrogram to 128x128

spectrograms = []
labels = []

for index, row in df.iterrows():
    file_path = os.path.join(
        "UrbanSound8K", "audio",
        "fold" + str(row["fold"]),
        row["slice_file_name"]
    )

    try:
        y, sr = librosa.load(file_path, duration=4.0)

        # Compute Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_SIZE)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize to fixed shape 128x128
        mel_resized = tf.image.resize(mel_db[..., np.newaxis], [IMG_SIZE, IMG_SIZE]).numpy()

        spectrograms.append(mel_resized)
        labels.append(row["class"])

        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1} / {len(df)} files...")

    except Exception as e:
        print(f"Error: {e}")

# ---- Step 2: Prepare data ----
X = np.array(spectrograms)   # shape: (8732, 128, 128, 1)
le = LabelEncoder()
y = le.fit_transform(np.array(labels))
class_names = le.classes_

print("\nSpectrogram data shape:", X.shape)
print("Number of classes:", len(class_names))

# Normalize pixel values to 0-1
X = (X - X.min()) / (X.max() - X.min())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---- Step 3: Build the CNN ----
print("\nBuilding CNN...")

model = keras.Sequential([
    # First conv block
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 1)),
    keras.layers.MaxPooling2D(2, 2),

    # Second conv block
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),

    # Third conv block
    keras.layers.Conv2D(128, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),

    # Flatten and classify
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),   # prevents overfitting
    keras.layers.Dense(len(class_names), activation="softmax")  # 10 output classes
])

model.summary()

# ---- Step 4: Train the CNN ----
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining CNN...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ---- Step 5: Evaluate ----
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test Accuracy: {test_acc*100:.1f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report — CNN")
print(classification_report(y_test, y_pred, target_names=class_names))

# ---- Step 6: Save confusion matrix ----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — CNN Model")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_cnn.png")
print("CNN confusion matrix saved!")

# ---- Step 7: Plot training history ----
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("CNN Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("cnn_training_history.png")
print("Training history saved!")

# Save model for later
model.save("urban_sound_cnn.keras")
print("Model saved as urban_sound_cnn.keras")