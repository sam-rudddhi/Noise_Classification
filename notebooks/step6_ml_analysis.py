import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load features ----
df = pd.read_csv("features_mfcc.csv")
X = df.drop(columns=["label", "fold"]).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Train SVM on full data and get predictions ----
print("Training SVM for detailed analysis...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_preds = np.zeros(len(y_encoded), dtype=int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_encoded)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(X_train, y_train)
    all_preds[test_idx] = svm.predict(X_test)
    print(f"  Fold {fold+1} done")

# ---- Classification Report ----
print("\n" + "="*60)
print("CLASSIFICATION REPORT — SVM")
print("="*60)
print(classification_report(y_encoded, all_preds, target_names=class_names))

# ---- Confusion Matrix ----
cm = confusion_matrix(y_encoded, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix — SVM Model")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_svm.png")
print("Confusion matrix saved as confusion_matrix_svm.png")

# ---- Find most confused pairs ----
print("\nMost confused class pairs (top 5):")
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)  # ignore correct predictions

for _ in range(5):
    idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
    actual = class_names[idx[0]]
    predicted = class_names[idx[1]]
    count = cm_no_diag[idx]
    print(f"  {actual} → predicted as {predicted}: {count} times")
    cm_no_diag[idx] = 0