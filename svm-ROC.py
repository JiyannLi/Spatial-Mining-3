import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

#
try:
    iris = pd.read_csv("iris.csv")
except FileNotFoundError:
    iris = pd.read_excel("iris.xlsx")

print("✅ Data loaded successfully, first five rows:")
print(iris.head())

#  Data Preparation
X = iris.drop("species", axis=1)
y = iris["species"]

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Binarize labels for multi-class ROC
y_binarized = label_binarize(y_encoded, classes=[0, 1, 2])
n_classes = y_binarized.shape[1]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binarized, test_size=0.3, random_state=42
)

# Train SVM Models
kernels = ["linear", "poly", "rbf"]
colors = ["red", "green", "blue"]
svm_models = {}

for kernel in kernels:
    clf = OneVsRestClassifier(SVC(kernel=kernel, probability=True, random_state=42))
    clf.fit(X_train, y_train)
    svm_models[kernel] = clf

# Plot ROC Curves

plt.figure(figsize=(8, 6))

for kernel, color in zip(kernels, colors):
    clf = svm_models[kernel]
    y_score = clf.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    plt.plot(
        all_fpr,
        mean_tpr,
        color=color,
        label=f"{kernel} kernel (AUC = {np.mean(list(roc_auc.values())):.2f})",
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for SVM Kernels Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("SVM_ROC_Comparison.png", dpi=300)
plt.show()
