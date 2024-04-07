from ReconhecimentoDePadroes.KNN_DMC.basic import neighborhoods


class Dmc:
    train_data = []
    train_data_class = []
    centroids = []

    def train(self, data, train_data_class):
        self.train_data = data
        self.train_data_class = train_data_class
        self.calculate_centroids()

    def nn(self, vector, data):
        distances = neighborhoods(vector, data)
        position = distances.index(min(distances))
        return [data[position][-1]]

    def calculate_centroids(self):
        classes = set([target[0] for target in self.train_data_class])
        centroids = []
        for i in classes:
            centroid = [0 for _ in self.train_data[0]]
            n = 0
            for index, k in enumerate(self.train_data):
                if self.train_data_class[index][0] == i:
                    n += 1
                    for w in range(len(k)):
                        centroid[w] += k[w]
            for z, j in enumerate(centroid):
                centroid[z] = j/n
            centroid.append(i)
            centroids.append(centroid)
        self.centroids = centroids

    def predict(self, vector):
        return self.nn(vector, self.centroids)
