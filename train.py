import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_split=4)
model.fit(X, y)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

# plt.figure(figsize=(15, 10))
# tree.plot_tree(model, filled=True)
# plt.show()

joblib.dump(model, 'model.joblib')

# plt.imshow(data.images[5])
# plt.show()