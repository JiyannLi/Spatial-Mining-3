import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


try:
    iris = pd.read_csv("iris.csv")
except FileNotFoundError:
    iris = pd.read_excel("iris.xlsx")

X = iris.drop("species", axis=1)
y = iris["species"]

签
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 拆分
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 模型配置
configs = [
    {"name": "CART_no_pruning", "criterion": "gini"},
    {"name": "CART_depth3", "criterion": "gini", "max_depth": 3},
    {"name": "CART_depth7", "criterion": "gini", "max_depth": 7},
    {"name": "CART_ccp_alpha_0.01", "criterion": "gini", "ccp_alpha": 0.01},
    {"name": "C4.5_like", "criterion": "entropy", "max_features": "sqrt"},
    {"name": "C4.5_depth3", "criterion": "entropy", "max_features": "sqrt", "max_depth": 3},
    {"name": "C4.5_depth7", "criterion": "entropy", "max_features": "sqrt", "max_depth": 7},
]

# 结果记录
results = []

for cfg in configs:
    model_params = {k: v for k, v in cfg.items() if k != "name"}
    clf = DecisionTreeClassifier(**model_params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": cfg["name"],
        "Accuracy": round(acc, 3),
        "Tree Depth": clf.get_depth(),
        "Leaf Nodes": clf.get_n_leaves(),
        "Confusion Matrix": cm
    })

#
print(f"{'Model':<25}{'Accuracy':<10}{'Tree Depth':<12}{'Leaf Nodes':<12}")
for r in results:
    print(f"{r['Model']:<25}{r['Accuracy']:<10}{r['Tree Depth']:<12}{r['Leaf Nodes']:<12}")
