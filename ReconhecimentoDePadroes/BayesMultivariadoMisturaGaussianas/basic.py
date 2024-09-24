import math
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from ReconhecimentoDePadroes.BayesMultivariadoMisturaGaussianas.bayesDiscriminantMixtureGaussian import BayesDiscriminantMixtureGaussianClassifier

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Ajuste da função para plotar as distribuições gaussianas das classes
def plot_gaussian_surfaces(method, X, y, yLabel, df_collumns, startCollumn, stopCollumn):
    if isinstance(method, BayesDiscriminantMixtureGaussianClassifier):
        # Pega as médias e as matrizes de covariância das classes
        class_means, class_covs = method.get_mean_covs()

        # Define o número de pontos no meshgrid
        num_points = 1000
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Cria o meshgrid
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = np.linspace(y_min, y_max, num_points)
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)

        colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Ajustar o número de cores conforme o número de classes

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Itera sobre as classes para plotar as distribuições gaussianas
        for class_label in np.unique(y):
            means = class_means[class_label]
            covariances = class_covs[class_label]

            # Itera sobre os componentes da mistura de gaussianas para a classe atual
            for component_idx, (mean, covariance) in enumerate(zip(means, covariances)):
                # Calcula os valores da PDF para os pontos do meshgrid para o componente atual
                xy_mesh = np.column_stack([x_mesh.flatten(), y_mesh.flatten()])
                pdf_values = multivariate_normal.pdf(xy_mesh, mean=mean, cov=covariance)
                pdf_values = pdf_values.reshape(x_mesh.shape)

                # Evita plotar valores muito baixos
                pdf_values[pdf_values <= 0.00005] = np.nan

                # Plota a superfície da distribuição gaussiana para o componente atual
                ax.plot_surface(x_mesh, y_mesh, pdf_values, color=colors[class_label % len(colors)], alpha=0.3)

        # Plota os pontos de dados
        for class_label in np.unique(y):
            ax.scatter(X[y == class_label, 0], X[y == class_label, 1], np.zeros_like(X[y == class_label, 1]),
                       label=f'Classe {yLabel[class_label]}', c=colors[class_label % len(colors)])

        # Títulos e labels dos eixos
        ax.set_xlabel([item for item in df_collumns if 'Class' != item][startCollumn])
        ax.set_ylabel([item for item in df_collumns if 'Class' != item][stopCollumn - 1])
        ax.set_zlabel('Densidade')
        ax.set_title('Distribuição Gaussiana com Pontos')
        ax.legend()

        # Exibe o gráfico
        plt.show()


def run_test(df, df_collumns, method, startCollumn, stopCollumn, step=0.02, file="result.csv"):
    results = []
    datas = []

    for i in range(20):
        shuffled_df = df.sample(frac=1)
        numpy_array_data_data = shuffled_df.drop(columns=['Class']).values
        numpy_array_class_data = shuffled_df['Class'].values

        train_data = numpy_array_data_data[0: math.floor(len(numpy_array_data_data) * 0.8)]
        train_class_data = numpy_array_class_data[0: math.floor(len(numpy_array_data_data) * 0.8)]

        yLabel = list(set([target for target in train_class_data]))
        y = np.array([yLabel.index(target) for target in train_class_data])

        method.train(
            train_data,
            y
        )

        correct_count = 0
        count = 0

        for n in range(math.floor(len(numpy_array_data_data) * 0.8), len(numpy_array_data_data)):
            predicted_class = method.predict(numpy_array_data_data[n])
            if yLabel[predicted_class] == numpy_array_class_data[n]:
                correct_count += 1

            count += 1

        results.append(correct_count / count)
        datas.append(shuffled_df)

        if i == 19:
            # SAVE EXCEL
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            (shuffled_df[[df_collumns[startCollumn], df_collumns[stopCollumn-1], 'Class']]).to_excel(file, index=False)

            y = np.array([yLabel.index(target) for target in train_class_data])

            method.train(train_data[:, startCollumn:stopCollumn], y)
            X = train_data[:, startCollumn:stopCollumn]

            # Definir os limites do gráfico
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            # Criar uma grade de pontos para fazer a superfície de decisão
            xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

            # Calcular as classes para cada ponto do mesh grid
            Z = np.zeros(xx.shape)
            with tqdm(total=Z.shape[0]) as barra_progresso:
                for i in range(Z.shape[0]):
                    barra_progresso.update(1)
                    for j in range(Z.shape[1]):
                        Z[i, j] = method.predict([xx[i, j], yy[i, j]])

            # Plotar a superfície de decisão
            plt.contourf(xx, yy, Z, alpha=0.8)

            # Plotar os pontos de dados
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

            plt.xlabel([item for item in df_collumns if 'Class' != item][startCollumn])
            plt.ylabel([item for item in df_collumns if 'Class' != item][stopCollumn-1])
            plt.title('Superficie de decisão')

            # Adicionar legendas
            plt.legend(handles=scatter.legend_elements()[0], labels=yLabel, loc='upper right')

            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.show()

            # GAUSSIANA MULTIVALORADA
            if isinstance(method, BayesDiscriminantMixtureGaussianClassifier):
               plot_gaussian_surfaces(method, X, y, yLabel, df_collumns, startCollumn, stopCollumn)

    # Resultados
    print("Resultados: ", results)
    result_data = np.array(results)

    mediana = np.sort(result_data)[math.floor(len(result_data) / 2)]

    # Encontrar o índice da mediana
    indice_mediana = np.where(result_data == mediana)

    mediana_df = datas[indice_mediana[0][0]]
    numpy_array_data_data = mediana_df.drop(columns='Class').values
    numpy_array_class_data = mediana_df['Class'].values

    train_data = numpy_array_data_data[0: math.floor(len(numpy_array_data_data) * 0.8)]
    train_class_data = numpy_array_class_data[0: math.floor(len(numpy_array_data_data) * 0.8)]

    yLabel = list(set([target for target in train_class_data]))
    y = np.array([yLabel.index(target) for target in train_class_data])

    method.train(
        train_data,
        y
    )

    correct_count = 0
    count = 0

    confusion_matrix = np.zeros((len(yLabel), len(yLabel))).tolist()

    for n in range(math.floor(len(numpy_array_data_data) * 0.8), len(numpy_array_data_data)):
        predicted_class = method.predict(numpy_array_data_data[n])
        if yLabel[predicted_class] == numpy_array_class_data[n]:
            correct_count += 1

        confusion_matrix[yLabel.index(numpy_array_class_data[n])][predicted_class] += 1
        count += 1

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

    # Mostrar tabela
    print(tabulate(data, headers=headers, tablefmt="grid"))

    # Média
    mean = np.mean(result_data)
    # Desvio Padrão
    std_dev = np.std(result_data)

    print("Acurácia (Média):", mean)
    print("Desvio Padrão:", std_dev)

