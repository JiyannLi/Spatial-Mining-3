import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 读取+ 检查
try:
    iris = pd.read_excel("iris.xlsx")
except Exception as e:
    print("数据读取出错：", e)
    raise

# 显示前5行
print("数据前五行：")
print(iris.head())

# 特征与标签
X = iris.drop("species", axis=1)
y = iris["species"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 模型初始化
id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
c45 = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', random_state=42)
cart = DecisionTreeClassifier(criterion='gini', random_state=42)

# 拟合模型
id3.fit(X_train, y_train)
c45.fit(X_train, y_train)
cart.fit(X_train, y_train)

# 可视化图并保存
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)
plot_tree(id3, filled=True, feature_names=X.columns, class_names=le.classes_)
plt.title("ID3 Decision tree")

plt.subplot(1, 3, 2)
plot_tree(c45, filled=True, feature_names=X.columns, class_names=le.classes_)
plt.title("C4.5 Decision tree")

plt.subplot(1, 3, 3)
plot_tree(cart, filled=True, feature_names=X.columns, class_names=le.classes_)
plt.title("CART Decision tree")

plt.tight_layout()
plt.savefig("all_trees_combined.png", dpi=300)
plt.show()
# 保存
for model, name in zip([id3, c45, cart], ["ID3", "C4.5", "CART"]):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=le.classes_, ax=ax)
    ax.set_title(f"{name} Decision tree")
    fig.savefig(f"{name}_tree.png", dpi=300)
    plt.close(fig)

print(" 所有图像已保存为：ID3_tree.png、C4.5_tree.png、CART_tree.png 和 all_trees_combined.png")

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

models = {
    "ID3": id3,
    "C4.5": c45,
    "CART": cart
}

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=axs[i], cmap="Blues", values_format='d')
    axs[i].set_title(f"{name} Confusing matrix\nAccuracy: {acc:.2f}")

plt.tight_layout()
plt.savefig("decision_tree_confusion_matrices.png", dpi=300)
plt.show()

