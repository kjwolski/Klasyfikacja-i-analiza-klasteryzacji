from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("Bez skalowania")
# print(X_train)
# print("Ze skalowaniem")
# print(X_train_scaled)

k_range = range(1,31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

# print(scores)

plt.figure(figsize=(10,5))
plt.plot(k_range, scores, marker='o')
plt.grid(True)
plt.xlabel('Wartość k (odpytywana liczba sąsiadów)')
plt.ylabel('Dokładność')
plt.title('Dokładność kNN dla różnych wartości k (liczby sąsiadów)')
# plt.show()

optimal_k = k_range[np.argmax(scores)]
knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
knn_classifier.fit(X_test_scaled, y_test)
y_pred = knn_classifier.predict(X_test_scaled)
print(f"Dokładność KNN: {accuracy_score(y_test, y_pred)}")