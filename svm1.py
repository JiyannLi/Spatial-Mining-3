import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

#
try:
    iris = pd.read_csv("iris.csv")
except FileNotFoundError:
    iris = pd.read_excel("iris.xlsx")

print(" 数据读取成功，前五行：")
print(iris.head())

#
X = iris.drop("species", axis=1)
y = iris["species"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 降维（2D）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

#  定义 SVM
kernels = ["linear", "poly", "rbf"]
models = {}
accuracies = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train, y_train)
    models[kernel] = svm

# 分类边界可视化
xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min()-1, X_pca[:, 0].max()+1, 500),
                     np.linspace(X_pca[:, 1].min()-1, X_pca[:, 1].max()+1, 500))

for kernel in kernels:
    model = models[kernel]
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(f"SVM  - {kernel} stone")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(*scatter.legend_elements(), title="Classification")
    plt.tight_layout()
    plt.savefig(f"SVM1_boundary_{kernel}.png", dpi=300)
    plt.show()

# 精度评估与混淆矩阵
for kernel in kernels:
    model = models[kernel]
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[kernel] = acc

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title(f"{kernel} stone SVM1\nAccuracy: {acc:.2f}")
    plt.tight_layout()
    plt.savefig(f"SVM1_confusion_{kernel}.png", dpi=300)
    plt.show()

#  打印精度
for kernel, acc in accuracies.items():
    print(f"{kernel}  SVM1 Accuraccy：{acc:.2f}")