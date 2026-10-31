import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load features ----
print("Loading features...")
df = pd.read_csv("features_mfcc.csv")

# Separate features (X) and labels (y)
X = df.drop(columns=["label", "fold"]).values   # 40 MFCC columns
y = df["label"].values                            # class names

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Classes:", np.unique(y))
print()

# ---- Encode labels as numbers ----
# ML models need numbers, not text
# So "dog_bark" → 3, "car_horn" → 1, etc.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Label encoding:", list(zip(le.classes_, range(len(le.classes_)))))
print()

# ---- Scale features ----
# Standardize so all MFCC values are on similar scale
# This helps ML models perform better
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Train & Evaluate each model ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel="rbf", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    # 5-fold cross validation — trains and tests 5 times on different splits
    scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring="accuracy")
    
    avg_accuracy = np.mean(scores)
    results[model_name] = avg_accuracy
    
    print(f"  Accuracy scores: {scores.round(3)}")
    print(f"  Average accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
    print()

# ---- Summary ----
print("=" * 40)
print("SUMMARY — Model Comparison")
print("=" * 40)
for model_name, acc in results.items():
    print(f"{model_name:25s}: {acc*100:.1f}%")

# ---- Bar chart comparison ----
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), [v*100 for v in results.values()], color=["steelblue", "coral", "green"])
plt.title("ML Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
for i, (k, v) in enumerate(results.items()):
    plt.text(i, v*100 + 1, f"{v*100:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("ml_comparison.png")
print("\nChart saved as ml_comparison.png")