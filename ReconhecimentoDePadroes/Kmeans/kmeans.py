import math
from collections import Counter

import numpy as np
from ReconhecimentoDePadroes.Kmeans.basic import euclidean_distance


class Kmeans:
    def __init__(self, k=5, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.centroids_label = []
        self.clusters = []

    def initialize_centroids(self, data):
        random_indices = np.random.permutation(data.shape[0])
        centroids = data[random_indices[:self.k]]
        return centroids

    def create_clusters(self, data, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, point in enumerate(data):
            centroid_idx = self.closest_centroid(point, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closest_centroid(self, point, centroids):
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        return np.argmin(distances)

    def compute_centroids(self, data, clusters):
        centroids = np.zeros((self.k, data.shape[1]))
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_mean = np.mean(data[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
            else:
                centroids[cluster_idx] = data[0]
        return centroids

    def has_converged(self, old_centroids, new_centroids):
        distances = [euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def train(self, data, labels, recursive=False):
        if self.k is None or not recursive:
            self.find_best_k(data, labels)
        self.centroids = []
        self.centroids_label = []
        centroids = self.initialize_centroids(data)
        clusters = []

        for _ in range(self.max_iterations):
            clusters = self.create_clusters(data, centroids)
            old_centroids = centroids
            centroids = self.compute_centroids(data, clusters)

            if self.has_converged(old_centroids, centroids):
                break

        for centroid, cluster in zip(centroids, clusters):
            if not math.isnan(centroid[0]) and not math.isnan(centroid[1]):
                self.centroids.append(centroid)
                self.filter_and_find_majority(labels, cluster)

    def find_best_k(self, data, labels):
        best_k = 0
        best_accuracy = 0
        for k in range(2, 21):

            if len(data) % 5 > 0:
                data_parts = np.split(data[0:len(data) - len(data) % 5], 5)
                data_class_parts = np.split(labels[0:len(data) - len(data) % 5], 5)

                data_parts[4] = np.concatenate((data_parts[4], data[:(- len(data) % 5)]))
                data_class_parts[4] = np.concatenate((data_class_parts[4], labels[:(- len(data) % 5)]))
            else:
                data_parts = np.split(data, 5)
                data_class_parts = np.split(labels, 5)

            accuracy_vector = []
            for i, part in enumerate(data_parts):
                train_data_parts = None
                train_data_class_parts = None
                for j, part2 in enumerate(data_parts):
                    if j != i:
                        if train_data_parts is None:
                            train_data_parts = part2.copy()
                            train_data_class_parts = data_class_parts[j].copy()
                        else:
                            train_data_parts = np.concatenate((train_data_parts, part2))
                            train_data_class_parts = np.concatenate((train_data_class_parts, data_class_parts[j]))

                self.k = k
                self.train(train_data_parts, train_data_class_parts, True)

                count = 0
                correct = 0
                for index, vector in enumerate(part):
                    if self.predict(vector) == data_class_parts[i][index]:
                        correct += 1
                    count += 1

                accuracy_vector.append(correct/count)

            np_accucary_vector = np.array(accuracy_vector)

            if np_accucary_vector.mean() >= best_accuracy:
                best_accuracy = np_accucary_vector.mean()
                best_k = k

        # print("Escolhido:", best_accuracy, best_k)
        self.k = best_k

    def filter_and_find_majority(self, labels, indices):
        filtered_labels = labels[indices]
        label_counts = Counter(filtered_labels)
        if len(label_counts) == 0:
            majority_label = labels[0]
        else:
            majority_label = label_counts.most_common(1)[0][0]

        self.centroids_label.append(majority_label)

    def predict(self, vector):
        centroid_idx = self.closest_centroid(vector, self.centroids)
        return self.centroids_label[centroid_idx]

