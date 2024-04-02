import numpy as np
import pandas as pd
from ReconhecimentoDePadroes.KNN_DMC.basic import euclidean_distance, majority, run_test


def find_index(data, value):
    for index, i in enumerate(data):
        if len(i) == len(value) and len(i) == sum([1 for x, j in zip(i, value) if x == j]):
            return index


class Knn:
    k = 5
    train_data = []
    train_data_class = []

    def __init__(self, k=5, train_data=None, train_data_class=None):
        self.k = k
        self.train_data = train_data
        self.train_data_class = train_data_class

    def train(self, data, train_data_class):
        self.find_best_k(data, train_data_class)

        self.train_data = data
        self.train_data_class = train_data_class

    def find_best_k(self, data, train_data_class):
        best_k = 0
        best_accuracy = 0
        for k in range(2, 22):

            if len(data) % 5 > 0:
                data_parts = np.split(data[0:len(data) - len(data) % 5], 5)
                data_class_parts = np.split(train_data_class[0:len(data) - len(data) % 5], 5)

                data_parts[4] = np.concatenate((data_parts[4], data[:(- len(data) % 5)]))
                data_class_parts[4] = np.concatenate((data_class_parts[4], train_data_class[:(- len(data) % 5)]))
            else:
                data_parts = np.split(data, 5)
                data_class_parts = np.split(train_data_class, 5)

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
                self.train_data = train_data_parts
                self.train_data_class = train_data_class_parts
                self.k = k

                count = 0
                correct = 0
                for index, vector in enumerate(part):
                    if self.predict(vector)[0] == data_class_parts[i][index][0]:
                        correct += 1
                    count += 1

                accuracy_vector.append(correct/count)

            np_accucary_vector = np.array(accuracy_vector)

            # print(np_accucary_vector.mean(), k)
            if np_accucary_vector.mean() > best_accuracy:
                best_accuracy = np_accucary_vector.mean()
                best_k = k

        # print("Escolhido:", best_accuracy, best_k)
        self.k = best_k

    def predict(self, vector):
        new_data = sorted(range(len(self.train_data)), key=lambda tup: euclidean_distance(self.train_data[tup], vector))
        classes = dict()
        for i in range(self.k):
            if self.train_data_class[new_data[i]][0] in classes.keys():
                classes[self.train_data_class[new_data[i]][0]] += 1
            else:
                classes[self.train_data_class[new_data[i]][0]] = 1
        return majority(classes)


if __name__ == '__main__':
    method = Knn(5)

    iris_collumns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
    df_iris = pd.read_csv('./datasets/iris.data', names=iris_collumns, delimiter=',')

    print("Running KNN - Iris")
    run_test(df_iris, iris_collumns, method, 0, 2, isKnn=True)

    artificial_collumns = ['Feature 1', 'Feature 2', 'Class']
    df_artificial = pd.read_csv('./datasets/artificial.data', names=artificial_collumns, delimiter=',')

    print("Running KNN - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, isKnn=True)

    collumn_collumns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Class']
    df_column = pd.read_csv('./datasets/column_3C.dat', names=collumn_collumns, delimiter=' ')

    print("Running KNN - Collumn")
    run_test(df_column, collumn_collumns, method, 0, 2, isKnn=True)
