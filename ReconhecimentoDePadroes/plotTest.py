import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # We'll use only the first two features for visualization
y = iris.target

# Instantiate the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Choose the number of neighbors here

# Train the KNN classifier
knn.fit(X, y)

# Define plot boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Define the step size for the mesh grid
step = 0.1

# Create a mesh grid to plot the decision surface
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

# Predict the classes for each point in the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision surface
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

# Add labels to the plot
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Decision Surface using KNN (k=5) on Iris Dataset')

# Add a legend indicating the class labels
plt.legend(handles=[plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, label=label) for label, color in zip(iris.target_names, plt.cm.Paired.colors)], loc='upper right')

plt.show()
