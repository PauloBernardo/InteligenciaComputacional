import random

import math
from copy import deepcopy
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd

dataset_test = [
    ['esferico', 'sim', 'alaranjado', 'rugosa', 'nao', 'LARANJA'],
    ['esferico', 'nao', 'vermelho', 'lisa', 'nao', 'MACA'],
    ['esferico', 'sim', 'alaranjado', 'rugosa', 'sim', 'TANGERINA']
]


def encode(actual_value, possible_values):
    s = list(set(sorted(possible_values)))
    s.sort()
    return s.index(actual_value)


def create_domain(dataset):
    number_of_columns = len(dataset[0]) - 1
    number_of_rows = len(dataset)
    domains = []
    for i in range(number_of_columns):
        domain = []
        for j in range(number_of_rows):
            if (dataset[j][i] not in domain) and isinstance(dataset[j][i], str):
                domain.append(dataset[j][i])
        domain.sort()
        if domain not in domains:
            domains.append(domain)
        domains.sort()
    return domains


def search_row(string, domain):
    for i in domain:
        if string in i:
            return i
    return -1


def transform_dataset(dataset):
    number_of_columns = len(dataset[0]) - 1
    number_of_rows = len(dataset)
    domains = create_domain(dataset)
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            row = search_row(dataset[i][j], domains)
            dataset[i][j] = encode(dataset[i][j], row)
    return dataset


def euclidean_distance(a, b):
    d = 0
    for i, j in zip(a, b):
        d += (i - j) ** 2
    return math.sqrt(d)


def transpose(matrix):
    n = len(matrix)
    m = len(matrix[0])
    new_matrix = []
    for i in range(m):
        l = []
        for j in range(n):
            l.append(matrix[j][i])
        new_matrix.append(l)
    return new_matrix


def new_label(dataset, labels):
    dataset = transpose(dataset)
    new_labels = []
    for i, j in zip(labels, dataset[:-1]):
        new_l = set(j)
        if "sim" in new_l or "nao" in new_l:
            new_labels.append(i)
        else:
            for k in new_l:
                new_labels.append(k)
    return new_labels


def one_hot_encoding(dataset, labels):
    new_dataset = []
    new_labels = new_label(dataset, labels)
    for i in range(len(dataset)):
        l = []
        for j in range(len(new_labels) + 1):
            l.append(0)
        new_dataset.append(l)
    for i in range(len(dataset)):
        for j in range(len(dataset[0])-1):
            try:
                col = new_labels.index(labels[j])
                if dataset[i][j].lower() == 'nao':
                    new_value = 0
                elif dataset_test[i][j].lower() == 'sim':
                    new_value = 1
                else:
                    new_value = dataset[i][j]

                new_dataset[i][col] = new_value
            except ValueError as e:
                print(e)
                col = new_labels.index(dataset[i][j])
                new_dataset[i][col] = 1
        new_dataset[i][-1] = dataset[i][-1]
    return new_labels, new_dataset


def normalize(matrix, new_min, new_max):
    t = transpose(matrix)
    for i, l in enumerate(t):
        mi = min(l)
        mx = max(l)
        for j, c in enumerate(l):
            t[i][j] = (t[i][j] - mi) / (mx - mi) * (new_min - new_max) + new_min
    return transpose(t)


def neighborhoods(vector, table):
    neighbors = []
    for i in table:
        neighbors.append(euclidean_distance(vector, i[:-1]))
    return neighbors


def majority(dictionary):
    return max(dictionary.items(), key=lambda x: x[1])


def calculate_centroids(data, target):
    classes = set(target)
    centroids = []
    for i in classes:
        centroid = [0 for _ in data[0]]
        n = 0
        for index, k in enumerate(data):
            if target[index] == i:
                n += 1
                for w in range(len(k)):
                    centroid[w] += k[w]
        for z, j in enumerate(centroid):
            centroid[z] = j/n
        centroid.append(i)
        centroids.append(centroid)
    return centroids


if __name__ == '__main__':
    print(encode('sim', ['sim', 'nao']))
    print(encode('nao', ['sim', 'nao']))
    print(encode('oval', ['esferico', 'oval', 'alongado']))
    print(create_domain(dataset_test))
    print(search_row('alaranjado', create_domain(dataset_test)))
    nDs = transform_dataset(deepcopy(dataset_test))
    print(nDs)
    print(euclidean_distance(nDs[0][:-1], nDs[1][:-1]))
    print(euclidean_distance(nDs[0][:-1], nDs[2][:-1]))
    print(euclidean_distance(nDs[1][:-1], nDs[2][:-1]))


    le = preprocessing.LabelEncoder()
    le.fit(['esferico', 'oval', 'alongado'])
    print(le.transform(['oval', 'oval', 'esferico']))

    labels = 'Formato', 'Fruta Citrica', 'Cor', 'Rugosidade', 'Cheiro'
    print(one_hot_encoding(dataset_test, labels))

    X = [i[:-1] for i in nDs]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X)

    print(enc.transform(X).toarray())

    df = pd.DataFrame(data=[i[:-1] for i in dataset_test], columns=labels)
    print(df.head())
    print(pd.get_dummies(df))

    scalier = MinMaxScaler()
    random_matrix = [
        [random.random() for i in range(10)],
        [random.random() for i in range(10)],
        [random.random() for i in range(10)]
    ]
    scalier.fit(random_matrix)
    MinMaxScaler(copy=True, feature_range=(0, 1))
    print(scalier.transform(random_matrix))
