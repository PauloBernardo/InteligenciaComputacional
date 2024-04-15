import numpy as np


class BayesClassifier:
    train_data = []
    train_data_class = []

    def __init__(self,  regularization=1e-6):
        self.class_prior = None
        self.class_mean = None
        self.class_covariance = None
        self.classes = None
        self.regularization = regularization

    def train(self, X, y, class_prior=None):
        self.train_data = X
        self.train_data_class = y
        self.classes = np.unique(y)
        self.class_prior = class_prior if class_prior is not None else np.zeros(len(self.classes))
        self.class_mean = {}
        self.class_covariance = {}

        for i, c in enumerate(self.classes):
            X_c = self.train_data[y == c]
            if class_prior is None:
                self.class_prior[i] = len(X_c) / len(X)
            self.class_mean[c] = np.mean(X_c, axis=0)
            self.class_covariance[c] = np.cov(X_c, rowvar=False) + self.regularization * np.identity(self.train_data.shape[1])

    def predict(self, x):
        class_scores = []
        for c in self.classes:
            # Calculando a probabilidade da classe usando a distribuição multivariada normal
            class_mean = self.class_mean[c]
            class_covariance = self.class_covariance[c]
            class_prior = self.class_prior[c]
            # Calcular a probabilidade da classe c usando a função de densidade gaussiana
            class_score = class_prior * self.gaussian_density(x, class_mean, class_covariance)
            class_scores.append(class_score)
        predicted_class = self.classes[np.argmax(class_scores)]
        return predicted_class

    def gaussian_density(self, x, class_mean, class_covariance):
        # Calcula a exponencial do termo dentro da PDF
        exponent = -0.5 * (x - class_mean).T @ np.linalg.inv(class_covariance) @ (x - class_mean)
        # Calcula a constante de normalização
        norm_const = 1 / ((2 * np.pi) ** (len(self.classes) / 2) * np.linalg.det(class_covariance) ** 0.5)
        return norm_const * np.exp(exponent)

    def get_mean_covs(self):
        return [self.class_mean[c] for i, c in enumerate(self.classes)], [self.class_covariance[c] for i, c in enumerate(self.classes)]
