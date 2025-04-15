import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("titanic.csv")

# Drop missing Age
df = df.dropna(subset=["Age"])

# Encode 'Sex'
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Features and label
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)
print(f"Decision Tree Accuracy: {tree_acc * 100:.2f}%")

# Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)
forest_acc = accuracy_score(y_test, forest_pred)
print(f"Random Forest Accuracy: {forest_acc * 100:.2f}%")

# Feature importance
importances = forest.feature_importances_
features = X.columns

print("\nRandom Forest Feature Importances:")
for feat, score in zip(features, importances):
    print(f"{feat}: {score:.4f}")
