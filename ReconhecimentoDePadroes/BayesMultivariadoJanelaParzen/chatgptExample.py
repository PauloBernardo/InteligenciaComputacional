import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar e treinar o classificador de discriminante linear
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Inicializar e treinar o classificador de discriminante quadrático
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Avaliar a precisão dos classificadores
lda_accuracy = accuracy_score(y_test, lda.predict(X_test))
qda_accuracy = accuracy_score(y_test, qda.predict(X_test))

print("Accuracy of Linear Discriminant Analysis:", lda_accuracy)
print("Accuracy of Quadratic Discriminant Analysis:", qda_accuracy)
