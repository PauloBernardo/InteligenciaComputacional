from ReconhecimentoDePadroes.KNN_DMC.basic import run_test
from ReconhecimentoDePadroes.KNN_DMC.dmc import Dmc
from ReconhecimentoDePadroes.KNN_DMC.knn import Knn
import pandas as pd


def run_knn():
    method = Knn(5)

    # iris_collumns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
    # df_iris = pd.read_csv('./datasets/iris.data', names=iris_collumns, delimiter=',')
    #
    # print("Running KNN - Iris")
    # run_test(df_iris, iris_collumns, method, 0, 2, isKnn=True, file="KNN_IRIS.xlsx")

    artificial_collumns = ['Feature 1', 'Feature 2', 'Class']
    df_artificial = pd.read_csv('../BayesMultivariado/datasets/artificial.data', names=artificial_collumns, delimiter=',')

    print(df_artificial)
    print("Running KNN - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, isKnn=True, file="KNN_ARTIFICIAL.xlsx")

    # collumn_collumns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Class']
    # df_column = pd.read_csv('./datasets/column_3C.dat', names=collumn_collumns, delimiter=' ')
    #
    # print("Running KNN - Collumn")
    # run_test(df_column, collumn_collumns, method, 0, 2, isKnn=True, file="KNN_COLLUMN.xlsx")


def run_dmc():
    method = Dmc()

    # iris_collumns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
    # df_iris = pd.read_csv('./datasets/iris.data', names=iris_collumns, delimiter=',')
    #
    # print("Running DMC - Iris")
    # run_test(df_iris, iris_collumns, method, 2, 4, file="DMC_IRIS.xlsx")

    artificial_collumns = ['Feature 1', 'Feature 2', 'Class']
    df_artificial = pd.read_csv('../BayesMultivariado/datasets/artificial.data', names=artificial_collumns, delimiter=',')

    print("Running DMC - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, file="DMC_ARTIFICIAL.xlsx")

    # collumn_collumns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Class']
    # df_column = pd.read_csv('./datasets/column_3C.dat', names=collumn_collumns, delimiter=' ')
    #
    # print("Running DMC - Collumn")
    # run_test(df_column, collumn_collumns, method, 0, 2, file="DMC_COLLUMN.xlsx")


if __name__ == '__main__':
    run_knn()
    run_dmc()
