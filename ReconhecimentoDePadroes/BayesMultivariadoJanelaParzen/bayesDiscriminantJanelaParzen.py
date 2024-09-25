import numpy as np

class ParzenWindow:
    def __init__(self, bandwidth=1.0):
        """
        Inicializa o estimador de densidade com Janela de Parzen.

        :param bandwidth: largura da janela ou bandwidth do kernel gaussiano.
        """
        self.bandwidth = bandwidth
        self.samples = None

    def fit(self, X):
        """
        Ajusta o estimador aos dados de amostra.

        :param X: array de amostras (n amostras x m features).
        """
        self.samples = X

    def score_samples(self, X):
        """
        Calcula a densidade de probabilidade para cada ponto em X.

        :param X: array de pontos para calcular a densidade (n amostras x m features).
        :return: densidades calculadas para cada ponto de X.
        """
        densities = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            densities[i] = np.mean([self._gaussian_kernel(x, sample) for sample in self.samples])
        return densities

    def _gaussian_kernel(self, x, xi):
        """
        Calcula o valor do kernel Gaussiano entre dois pontos.

        :param x: ponto de teste.
        :param xi: ponto de amostra.
        :return: valor do kernel.
        """
        d = len(x)
        coeff = 1 / (np.sqrt((2 * np.pi) ** d) * (self.bandwidth ** d))
        exponent = -0.5 * np.sum(((x - xi) / self.bandwidth) ** 2)
        return coeff * np.exp(exponent)


class BayesDiscriminantJanelaParzenClassifier:
    def __init__(self, bandwidth=1.0):
        """
        Inicializa o classificador Bayesiano com Janela de Parzen.

        :param bandwidth: largura da janela do kernel.
        """
        self.bandwidth = bandwidth
        self.class_densities = {}
        self.class_prior = {}
        self.classes = None

    def train(self, X, y):
        """
        Treina o classificador com dados de entrada.

        :param X: dados de entrada (n amostras x m features).
        :param y: rótulos das classes correspondentes.
        """
        self.classes = np.unique(y)
        self.class_prior = {c: np.mean(y == c) for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            density_estimator = ParzenWindow(bandwidth=self.bandwidth)
            density_estimator.fit(X_c)
            self.class_densities[c] = density_estimator

    def predict(self, x):
        """
        Prediz a classe de um novo ponto.

        :param x: ponto a ser classificado.
        :return: classe predita.
        """
        class_scores = []
        for c in self.classes:
            density_estimator = self.class_densities[c]
            log_likelihood = np.log(density_estimator.score_samples(np.array([x]))[0] + 1e-9)
            log_prior = np.log(self.class_prior[c])
            class_score = log_likelihood + log_prior
            class_scores.append(class_score)

        predicted_class = self.classes[np.argmax(class_scores)]
        return predicted_class


# Exemplo de uso
# Gerar dados de exemplo
# np.random.seed(42)
# X_class0 = np.random.normal(loc=0, scale=1, size=(50, 2))
# X_class1 = np.random.normal(loc=3, scale=1, size=(50, 2))
# X = np.vstack((X_class0, X_class1))
# y = np.array([0] * 50 + [1] * 50)
#
# # Treinar o classificador com janela de Parzen
# classifier = BayesParzenClassifier(bandwidth=0.5)
# classifier.train(X, y)
#
# # Predizer uma nova amostra
# sample = [1, 1]
# predicted_class = classifier.predict(sample)
# print(f"A amostra {sample} foi classificada como classe {predicted_class}.")
