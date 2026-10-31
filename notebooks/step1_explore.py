import pandas as pd
import matplotlib.pyplot as plt

# ---- Load the metadata CSV ----
df = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")

# ---- Basic info ----
print("Total audio files:", len(df))
print()
print("First 5 rows:")
print(df.head())
print()
print("Column names:", df.columns.tolist())
print()

# ---- How many clips per class? ----
print("Clips per class:")
print(df["class"].value_counts())

# ---- Plot a bar chart of class distribution ----
plt.figure(figsize=(12, 5))
df["class"].value_counts().plot(kind="bar", color="steelblue")
plt.title("Number of Audio Clips per Class")
plt.xlabel("Sound Class")
plt.ylabel("Number of Clips")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("class_distribution.png")
print()
print("Chart saved as class_distribution.png")