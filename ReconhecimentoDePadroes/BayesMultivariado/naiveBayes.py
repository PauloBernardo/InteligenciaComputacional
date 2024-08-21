import numpy as np


class NaiveBayesClassifier:
    train_data = []
    train_data_class = []

    def __init__(self,  regularization=1e-6):
        self.class_prior = None
        self.classes = None
        self.class_probabilities = None
        self.feature_probabilities = None
        self.regularization = regularization

    def train(self, X, y, class_prior=None):
        self.train_data = X
        self.train_data_class = y
        self.classes = np.unique(y)
        self.class_probabilities = class_prior if class_prior is not None else np.zeros(len(self.classes))
        self.feature_probabilities = None

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            if class_prior is None:
                self.class_probabilities[i] = len(X_c) / len(X)

            # Calcular a média e o desvio padrão de cada atributo para a classe c
            feature_means = np.mean(X_c, axis=0)
            feature_stddevs = np.std(X_c, axis=0)

            if self.feature_probabilities is None:
                self.feature_probabilities = [(feature_means, feature_stddevs)]
            else:
                self.feature_probabilities.append((feature_means, feature_stddevs))

    def predict(self, x):
        class_scores = []
        for i, c in enumerate(self.classes):
            # Calcular a probabilidade da classe c usando a função de densidade gaussiana
            class_score = np.sum(np.log(self.class_probabilities[i]) +
                                 np.log(self.gaussian_density(x, i)))
            class_scores.append(class_score)
        predicted_class = self.classes[np.argmax(class_scores)]
        return predicted_class

    def gaussian_density(self, x, class_index):
        means, stddevs = self.feature_probabilities[class_index]
        stddevs += self.regularization  # Adicionar regularização aos desvios padrão
        exponent = -0.5 * ((x - means) / stddevs) ** 2
        density = np.exp(exponent) / (np.sqrt(2 * np.pi) * stddevs)
        return density

    def get_mesans_stddevs(self):
        return self.feature_probabilities

