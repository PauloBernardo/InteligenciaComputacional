import numpy as np


class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=1000, tol=1e-6, regularization=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.means = None
        self.covariances = None
        self.weights = None

    def fit(self, X):
        n_samples, n_features = X.shape

        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False) + self.regularization * np.eye(n_features)
                                     for _ in range(self.n_components)])
        self.weights = np.ones(self.n_components) / self.n_components

        log_likelihood_prev = -np.inf

        for iteration in range(self.max_iter):
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights[k] * self._gaussian_density(X, self.means[k], self.covariances[k])

            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            N_k = responsibilities.sum(axis=0)
            self.means = (responsibilities.T @ X) / N_k[:, np.newaxis]
            self.covariances = np.zeros((self.n_components, n_features, n_features))

            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / N_k[k]
                self.covariances[k] += self.regularization * np.eye(n_features)

            self.weights = N_k / n_samples

            log_likelihood = np.sum(np.log(responsibilities.sum(axis=1)))
            if np.abs(log_likelihood - log_likelihood_prev) < self.tol:
                break
            log_likelihood_prev = log_likelihood

    def _gaussian_density(self, X, mean, covariance):
        n = len(mean)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ np.linalg.inv(covariance) * diff, axis=1)
        denominator = np.sqrt((2 * np.pi) ** n * np.linalg.det(covariance))
        return np.exp(exponent) / (denominator + 1e-9)

    def score_samples(self, X):
        log_likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_likelihood[:, k] = np.log(self.weights[k] + 1e-9) + np.log(self._gaussian_density(X, self.means[k], self.covariances[k]) + 1e-9)
        return np.sum(log_likelihood, axis=1)


class BayesDiscriminantMixtureGaussianClassifier:
    def __init__(self, n_components=3, regularization=1e-6):
        self.n_components = n_components
        self.regularization = regularization
        self.class_gmms = {}
        self.class_prior = {}
        self.classes = None

    def train(self, X, y, class_prior=None):
        self.classes = np.unique(y)
        if class_prior is None:
            self.class_prior = {c: np.mean(y == c) for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            gmm = GaussianMixtureModel(n_components=self.n_components, regularization=self.regularization)
            gmm.fit(X_c)
            self.class_gmms[c] = gmm

    def predict(self, x):
        class_scores = []
        for c in self.classes:
            gmm = self.class_gmms[c]
            log_likelihood = gmm.score_samples(np.array(x).reshape(1, -1))[0]
            log_prior = np.log(self.class_prior[c])
            class_score = log_likelihood + log_prior
            class_scores.append(class_score)

        predicted_class = self.classes[np.argmax(class_scores)]
        return predicted_class

    def get_mean_covs(self):
        means = {c: gmm.means for c, gmm in self.class_gmms.items()}
        covariances = {c: gmm.covariances for c, gmm in self.class_gmms.items()}
        return means, covariances



