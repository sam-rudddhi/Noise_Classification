import librosa
import numpy as np
import matplotlib.pyplot as plt

# ---- Load audio ----
audio_path = "UrbanSound8K/audio/fold5/100032-3-0-0.wav"
y, sr = librosa.load(audio_path, duration=4.0)

# ---- Extract MFCCs ----
# MFCCs describe the "texture/shape" of the sound
# We extract 40 coefficients
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

print("MFCC shape:", mfccs.shape)
print("(40 coefficients x number of time steps)")
print()

# For ML models, we take the MEAN of each coefficient across time
# This gives us one single vector of 40 numbers per audio clip
mfccs_mean = np.mean(mfccs, axis=1)

print("After taking mean — shape:", mfccs_mean.shape)
print("This is our feature vector for ML models:")
print(mfccs_mean)

# ---- Visualize MFCCs ----
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis="time")
plt.colorbar()
plt.title("MFCCs — Dog Bark")
plt.ylabel("MFCC Coefficient")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig("mfcc_visualization.png")
print()
print("MFCC visualization saved!")