from sklearn.datasets import load_iris
from NN_KNN_DMC.basic import euclidean_distance, majority


def find_index(data, value):
    for index, i in enumerate(data):
        if len(i) == len(value) and len(i) == sum([1 for x, j in zip(i, value) if x == j]):
            return index


def knn(k, vector, data, target):
    labels = ['setosa', 'versicolor', 'virginica']
    new_data = sorted(data, key=lambda tup: euclidean_distance(tup, vector))
    data_list = data.tolist()
    classes = dict()
    for i in range(k):
        if labels[target[find_index(data_list, new_data[i])]] in classes.keys():
            classes[labels[target[find_index(data_list, new_data[i])]]] += 1
        else:
            classes[labels[target[find_index(data_list, new_data[i])]]] = 1
    return majority(classes)


if __name__ == '__main__':
    iris = load_iris()
    test_1 = [4.3, 3.0, 1.1, 2.1]
    test_2 = [6.7, 3.0, 5.2, 2.3]
    test_3 = [5.7, 2.6, 3.5, 1.0]

    print("Test with a setosa => ", knn(5, test_1, iris['data'], iris['target']))
    print("Test with a virginica => ", knn(5, test_2, iris['data'], iris['target']))
    print("Test with a versicolor => ", knn(5, test_3, iris['data'], iris['target']))

    # Change vector_1 to wrong format
    test_1 = [6.3, 5.0, 4.1, 2.1]
    print("[Should be wrong] Test with a setosa => ", knn(5, test_1, iris['data'], iris['target']))
