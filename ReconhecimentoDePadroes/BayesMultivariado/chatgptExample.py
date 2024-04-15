import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using only the first two features for visualization
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the mean and covariance matrix for each class
class_means = []
class_covs = []
for i in range(3):
    X_class = X_train[y_train == i]
    class_means.append(np.mean(X_class, axis=0))
    class_covs.append(np.cov(X_class.T))

# Plot the contour plots for each class
plt.figure(figsize=(15, 5))

colors = ['r', 'g', 'b']
classes = ['Class 0', 'Class 1', 'Class 2']

for i in range(3):
    plt.subplot(1, 3, i+1)

    # Plot the data points for class i
    plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=classes[i])

    # Plot the contour plot for the Gaussian distribution of class i
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    xy = np.column_stack([xx.flatten(), yy.flatten()])
    Z = multivariate_normal.pdf(xy, mean=class_means[i], cov=class_covs[i]).reshape(xx.shape)
    plt.contour(xx, yy, Z, alpha=0.5, cmap='coolwarm')

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title(f'Class {i} Gaussian Distribution')
    plt.legend()
    plt.colorbar(label='Class Probability')

plt.tight_layout()
plt.show()
