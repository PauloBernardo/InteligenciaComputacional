import math
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm


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


def run_test(df, df_collumns, method, startCollumn, stopCollumn, isKnn = False, file="result.csv"):
    results = []
    ks = np.array([])
    datas = []

    for i in range(20):
        shuffled_df = df.sample(frac=1)
        numpy_array_data_data = shuffled_df[df_collumns[0:-1]].values
        numpy_array_class_data = shuffled_df[df_collumns[-1:]].values

        train_data = numpy_array_data_data[0: math.floor(len(numpy_array_data_data) * 0.8)]
        train_class_data = numpy_array_class_data[0: math.floor(len(numpy_array_data_data) * 0.8)]

        method.train(
            train_data,
            train_class_data
        )

        correct_count = 0
        count = 0

        yLabel = list(set([target[0] for target in train_class_data]))

        for n in range(math.floor(len(numpy_array_data_data) * 0.8), len(numpy_array_data_data)):
            predicted_class = method.predict(numpy_array_data_data[n])[0]
            if predicted_class == numpy_array_class_data[n][0]:
                correct_count += 1

            count += 1

        results.append(correct_count / count)
        datas.append(shuffled_df)
        if isKnn:
            ks = np.append(ks, method.k)

        if i == 19:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            (shuffled_df[[df_collumns[startCollumn], df_collumns[stopCollumn-1], 'Class']]).to_csv(file, index=False)
            # print("Dados de treinamento: ", train_data, train_class_data)
            # print("Dados de teste: ", numpy_array_data_data[math.floor(len(numpy_array_data_data) * 0.8):], numpy_array_class_data[math.floor(len(numpy_array_data_data) * 0.8):])

            y = np.array([[yLabel.index(target[0])] for target in train_class_data])

            method.train(train_data[:, startCollumn:stopCollumn], y)
            X = train_data[:, startCollumn:stopCollumn]

            # Definir os limites do gráfico
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            # Criar uma grade de pontos para fazer a superfície de decisão
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

            # Calcular as classes para cada ponto do mesh grid
            Z = np.zeros(xx.shape)
            with tqdm(total=Z.shape[0]) as barra_progresso:
                # print(Z.shape[0])
                for i in range(Z.shape[0]):
                    barra_progresso.update(1)
                    for j in range(Z.shape[1]):
                        Z[i, j] = method.predict([xx[i, j], yy[i, j]])[0]

            # Plotar a superfície de decisão
            plt.contourf(xx, yy, Z, alpha=0.8)

            # Plotar os pontos de dados
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

            plt.xlabel(df_collumns[startCollumn])
            plt.ylabel(df_collumns[stopCollumn-1])
            plt.title('Superficie de decisão')

            # Adicionar legendas
            plt.legend(handles=scatter.legend_elements()[0], labels=yLabel, loc='upper right')

            plt.show()

    print(results)
    if isKnn:
        print(ks)
        print(statistics.mode(ks))
    result_data = np.array(results)

    # Calcular a mediana
    mediana = np.median(result_data)
    # Encontrar o índice da mediana
    indice_mediana = np.where(result_data == mediana)

    mediana_df = datas[indice_mediana[0][0]]
    numpy_array_data_data = mediana_df[df_collumns[0:-1]].values
    numpy_array_class_data = mediana_df[df_collumns[-1:]].values

    train_data = numpy_array_data_data[0: math.floor(len(numpy_array_data_data) * 0.8)]
    train_class_data = numpy_array_class_data[0: math.floor(len(numpy_array_data_data) * 0.8)]

    method.train(
        train_data,
        train_class_data
    )

    correct_count = 0
    count = 0

    yLabel = list(set([target[0] for target in train_class_data]))
    confusion_matrix = np.zeros((len(yLabel), len(yLabel))).tolist()

    for n in range(math.floor(len(numpy_array_data_data) * 0.8), len(numpy_array_data_data)):
        predicted_class = method.predict(numpy_array_data_data[n])[0]
        if predicted_class == numpy_array_class_data[n][0]:
            correct_count += 1

        confusion_matrix[yLabel.index(numpy_array_class_data[n][0])][yLabel.index(predicted_class)] += 1
        count += 1

    # print("Dados de treinamento: ", train_data, train_class_data)
    # print("Dados de teste: ", numpy_array_data_data[math.floor(len(numpy_array_data_data) * 0.8):], numpy_array_class_data[math.floor(len(numpy_array_data_data) * 0.8):])
    print("Matriz de confusão por classe")
    data = []

    aux_v = [""]
    aux_v.extend([label for label in yLabel])
    data.append(aux_v)
    for i, label in enumerate(yLabel):
        aux_v = [label]
        aux_v.extend([confusion_matrix[i][yLabel.index(label)] for label in yLabel])
        data.append(aux_v)

    # Table headers
    headers = ["Valores previstos" for _ in yLabel]

    # Print the table
    print(tabulate(data, headers=headers, tablefmt="grid"))

    # Média
    mean = np.mean(result_data)
    # Desvio Padrão
    std_dev = np.std(result_data)

    print("Acurácia (Média):", mean)
    print("Desvio Padrão:", std_dev)

