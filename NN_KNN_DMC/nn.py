from sklearn.datasets import load_iris
from NN_KNN_DMC.basic import neighborhoods


def nn(vector, data, target):
    labels = ['setosa', 'versicolor', 'virginica']
    distances = neighborhoods(vector, data)
    position = distances.index(min(distances))
    return labels[target[position]]


if __name__ == '__main__':
    iris = load_iris()
    test_1 = [4.3, 3.0, 1.1, 2.1]
    test_2 = [6.7, 3.0, 5.2, 2.3]
    test_3 = [5.7, 2.6, 3.5, 1.0]

    print("Test with a setosa => ", nn(test_1, iris['data'], iris['target']))
    print("Test with a virginica => ", nn(test_2, iris['data'], iris['target']))
    print("Test with a versicolor => ", nn(test_3, iris['data'], iris['target']))

    # Change vector_1 to wrong format
    test_1 = [6.3, 5.0, 4.1, 2.1]
    print("[Should be wrong] Test with a setosa => ", nn(test_1, iris['data'], iris['target']))
