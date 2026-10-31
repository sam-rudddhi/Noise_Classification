import librosa
import numpy as np
import pandas as pd
import os

# ---- Load metadata ----
df = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")
print("Starting feature extraction for", len(df), "files...")
print("This will take 10-20 minutes. Please wait!\n")

# ---- Lists to store results ----
features = []   # will hold 40 MFCC numbers for each clip
labels = []     # will hold the class name for each clip
folds = []      # will hold the fold number for each clip

# ---- Loop through every audio file ----
for index, row in df.iterrows():

    # Build the file path
    file_path = os.path.join(
        "UrbanSound8K", "audio",
        "fold" + str(row["fold"]),
        row["slice_file_name"]
    )

    try:
        # Load audio
        y, sr = librosa.load(file_path, duration=4.0)

        # Extract 40 MFCCs and take mean across time
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Store results
        features.append(mfccs_mean)
        labels.append(row["class"])
        folds.append(row["fold"])

        # Print progress every 500 files
        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1} / {len(df)} files...")

    except Exception as e:
        print(f"Error on {file_path}: {e}")

# ---- Save everything to a CSV file ----
print("\nSaving features to file...")

# Create column names: mfcc_1, mfcc_2, ... mfcc_40
feature_columns = [f"mfcc_{i+1}" for i in range(40)]

# Create a dataframe
features_df = pd.DataFrame(features, columns=feature_columns)
features_df["label"] = labels
features_df["fold"] = folds

# Save to CSV
features_df.to_csv("features_mfcc.csv", index=False)

print("Done! Features saved to features_mfcc.csv")
print("Shape:", features_df.shape)
print(features_df.head())