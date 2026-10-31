import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ---- Load one audio file ----
# We're loading a dog bark from fold5
audio_path = "UrbanSound8K/audio/fold5/100032-3-0-0.wav"

# librosa.load gives us:
# y = the actual sound as an array of numbers
# sr = sampling rate (how many numbers per second, usually 22050)
y, sr = librosa.load(audio_path, duration=4.0)

print("Audio loaded!")
print("Sampling rate:", sr, "Hz")
print("Number of samples:", len(y))
print("Duration:", len(y)/sr, "seconds")
print("First 10 values of the audio signal:", y[:10])

# ---- Plot 1: Waveform ----
# This shows how the sound amplitude changes over time
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)   # 2 rows, 1 column, first plot
librosa.display.waveshow(y, sr=sr, color="steelblue")
plt.title("Waveform — Dog Bark")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# ---- Plot 2: Mel Spectrogram ----
# This shows how frequencies change over time — this is what we feed into CNN!
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # convert to decibels

plt.subplot(2, 1, 2)   # second plot
librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram — Dog Bark")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Mel scale)")

plt.tight_layout()
plt.savefig("audio_visualization.png")
print()
print("Visualization saved as audio_visualization.png")