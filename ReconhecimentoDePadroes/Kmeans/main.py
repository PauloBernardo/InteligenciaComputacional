import pandas as pd

from ReconhecimentoDePadroes.Kmeans.basic import run_test
from ReconhecimentoDePadroes.Kmeans.kmeans import Kmeans

def replace_question_mark_with_mean(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    mean_value = df[column_name].mean()

    df[column_name] = df[column_name].fillna(mean_value)

    df[column_name] = df[column_name].astype(int)

    return df


def normalize_column(column):
    min_val = column.min()
    max_val = column.max()
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column


def get_iris_data():
    iris_collumns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
    df_iris = pd.read_csv('./datasets/iris.data', names=iris_collumns, delimiter=',')

    return iris_collumns, df_iris


def get_artificial_data():
    artificial_collumns = ['Feature 1', 'Feature 2', 'Class']
    df_artificial = pd.read_csv('./datasets/artificial.data', names=artificial_collumns, delimiter=',')

    return artificial_collumns, df_artificial


def get_collumn_data():
    collumn_collumns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Class']
    df_column = pd.read_csv('./datasets/column_3C.dat', names=collumn_collumns, delimiter=' ')

    return collumn_collumns, df_column


def run_k_means():
    method = Kmeans(None)

    iris_collumns, df_iris = get_iris_data()
    print("Running Kmeans - Iris")
    run_test(df_iris, iris_collumns, method, 0, 2, step=0.1, file="KMEANS_IRIS.xlsx")

    artificial_collumns, df_artificial = get_artificial_data()
    print("Running Kmeans - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, step=0.01, file="KMEANS_ARTIFICIAL.xlsx")

    collumn_collumns, df_column = get_collumn_data()
    print("Running Kmeans - Collumn")
    run_test(df_column, collumn_collumns, method, 0, 2, step=0.1, file="KMEANS_COLLUMN.xlsx")


if __name__ == '__main__':
    run_k_means()
