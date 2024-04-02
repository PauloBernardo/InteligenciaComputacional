import pandas as pd
from ReconhecimentoDePadroes.KNN_DMC.basic import run_test, neighborhoods


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


if __name__ == '__main__':
    method = Dmc()

    iris_collumns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
    df_iris = pd.read_csv('./datasets/iris.data', names=iris_collumns, delimiter=',')

    print("Running DMC - Iris")
    run_test(df_iris, iris_collumns, method, 2, 4, file="DMC_IRIS.csv")

    artificial_collumns = ['Feature 1', 'Feature 2', 'Class']
    df_artificial = pd.read_csv('./datasets/artificial.data', names=artificial_collumns, delimiter=',')

    print("Running DMC - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, file="DMC_ARTIFICIAL.csv")

    collumn_collumns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Class']
    df_column = pd.read_csv('./datasets/column_3C.dat', names=collumn_collumns, delimiter=' ')

    print("Running DMC - Collumn")
    run_test(df_column, collumn_collumns, method, 0, 2, file="DMC_COLLUMN.csv")
