import pandas as pd

from ReconhecimentoDePadroes.BayesMultivariado.basic import run_test
from ReconhecimentoDePadroes.BayesMultivariado.bayes import BayesClassifier
from ReconhecimentoDePadroes.BayesMultivariado.bayesDiscriminant import BayesDiscriminantClassifier
from ReconhecimentoDePadroes.BayesMultivariado.naiveBayes import NaiveBayesClassifier
from ReconhecimentoDePadroes.KNN_DMC.dmc import Dmc
from ReconhecimentoDePadroes.KNN_DMC.knn import Knn


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


def get_breast_data():
    breast_collumns = [
        'Class',
        'age',
        'menopause',
        'tumor-size',
        'inv-nodes',
        'node-caps',
        'deg-malig',
        'breast',
        'breast-quad',
        'irradiat'
    ]
    df = pd.read_csv('./datasets/breast-cancer.data', names=breast_collumns, delimiter=',')

    # Label Encoding
    label_encoding_map = {
        '10-19': 0, '20-29': 2, '30-39': 3, '40-49': 4, '50-59': 5, '60-69': 6, '70-79': 7, '80-89': 8, '90-99': 9,
        'lt40': 0, 'ge40': 1, 'premeno': 2,
        '0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9, '50-54': 10, '55-59': 11,
        '0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '18-20': 6, '21-23': 7, '24-26': 8, '27-29': 9, '30-32': 10, '33-35': 11, '36-39': 12,
        'yes': 0, 'no': 1,
        '1': 0, '2': 1, '3': 2,
        'left': 0, 'right': 1,
        'left_up': 0, 'left_low': 1, 'right_up': 2, 'right_low': 3, 'central': 4,
    }
    df_breast = df.applymap(lambda x: label_encoding_map.get(x) if x in label_encoding_map else x)

    df_breast = replace_question_mark_with_mean(df_breast, "node-caps")
    df_breast = replace_question_mark_with_mean(df_breast, "breast-quad")

    return breast_collumns, df_breast


def get_demartology_data():
    dermatology_collumns = [
        'erythema',
        'scaling',
        'definite_borders',
        'itching',
        'koebner_phenomenon',
        'polygonal_papules',
        'follicular_papules',
        'oral_mucosal_involvement',
        'knee_and_elbow_involvement',
        'scalp_involvement',
        'family_history',
        'melanin_incontinence',
        'eosinophils_in_the_infiltrate',
        'PNL_infiltrate',
        'fibrosis_of_the_papillary_dermis',
        'exocytosis',
        'acanthosis',
        'hyperkeratosis',
        'parakeratosis',
        'clubbing_of_the_rete_ridges',
        'elongation_of_the_rete_ridges',
        'thinning_of_the_suprapapillary_epidermis',
        'spongiform_pustule',
        'munro_microabcess',
        'focal_hypergranulosis',
        'disappearance_of_the_granular_layer',
        'vacuolisation_and_damage_of_basal_layer',
        'spongiosis',
        'saw-tooth_appearance_of_retes',
        'follicular_horn_plug',
        'perifollicular_parakeratosis',
        'inflammatory_monoluclear_inflitrate',
        'band_like_infiltrate',
        'age',
        'Class']
    df = pd.read_csv('./datasets/dermatology.data', names=dermatology_collumns, delimiter=',')

    df_dermatology = replace_question_mark_with_mean(df, "age")
    df_dermatology['age'] = normalize_column(df['age'])

    return dermatology_collumns, df_dermatology


def run_bayes_discriminant():
    method = BayesDiscriminantClassifier()

    iris_collumns, df_iris = get_iris_data()
    print("Running Bayes Discriminant LDA - Iris")
    run_test(df_iris, iris_collumns, method, 0, 2, file="LDA_BAYES_IRIS.xlsx")
    #
    # artificial_collumns, df_artificial = get_artificial_data()
    # print("Running Bayes Discriminant LDA - Artificial")
    # run_test(df_artificial, artificial_collumns, method, 0, 2, step=0.01, file="LDA_BAYES_ARTIFICIAL.xlsx")
    #
    # collumn_collumns, df_column = get_collumn_data()
    # print("Running Bayes Discriminant LDA - Collumn")
    # run_test(df_column, collumn_collumns, method, 0, 2, step=0.1, file="LDA_BAYES_COLLUMN.xlsx")
    #
    # breast_collumns, df_breast = get_breast_data()
    # print("Running Bayes Discriminant LDA - Breast")
    # run_test(df_breast, breast_collumns, method, 2, 4, file="LDA_BAYES_BREAST.xlsx")
    #
    # dermatology_collumns, df_dermatology = get_demartology_data()
    # print("Running Bayes Discriminant LDA - Dermatology")
    # run_test(df_dermatology, dermatology_collumns, method, 15, 17, file="LDA_BAYES_DERMATOLOGY.xlsx")

    # method = BayesDiscriminantClassifier(method='QDA')
    #
    # iris_collumns, df_iris = get_iris_data()
    # print("Running Bayes Discriminant QDA - Iris")
    # run_test(df_iris, iris_collumns, method, 0, 2, file="QDA_BAYES_IRIS.xlsx")

    # artificial_collumns, df_artificial = get_artificial_data()
    # print("Running Bayes Discriminant QDA - Artificial")
    # run_test(df_artificial, artificial_collumns, method, 0, 2, step=0.01, file="QDA_BAYES_ARTIFICIAL.xlsx")
    #
    # collumn_collumns, df_column = get_collumn_data()
    # print("Running Bayes Discriminant QDA - Collumn")
    # run_test(df_column, collumn_collumns, method, 0, 2, step=0.1, file="QDA_BAYES_COLLUMN.xlsx")
    #
    # breast_collumns, df_breast = get_breast_data()
    # print("Running Bayes Discriminant QDA - Breast")
    # run_test(df_breast, breast_collumns, method, 2, 4, file="QDA_BAYES_BREAST.xlsx")
    #
    # dermatology_collumns, df_dermatology = get_demartology_data()
    # print("Running Bayes Discriminant QDA - Dermatology")
    # run_test(df_dermatology, dermatology_collumns, method, 15, 17, file="QDA_BAYES_DERMATOLOGY.xlsx")


def run_naive_bayes():
    method = NaiveBayesClassifier()

    iris_collumns, df_iris = get_iris_data()
    print("Running Bayes - Iris")
    run_test(df_iris, iris_collumns, method, 0, 2, file="NAIVE_BAYES_IRIS.xlsx")

    artificial_collumns, df_artificial = get_artificial_data()
    print("Running Bayes - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, step=0.01, file="NAIVE_BAYES_ARTIFICIAL.xlsx")

    collumn_collumns, df_column = get_collumn_data()
    print("Running Bayes - Collumn")
    run_test(df_column, collumn_collumns, method, 0, 2, step=0.1, file="NAIVE_BAYES_COLLUMN.xlsx")

    breast_collumns, df_breast = get_breast_data()
    print("Running Bayes - Breast")
    run_test(df_breast, breast_collumns, method, 2, 4, file="NAIVE_BAYES_BREAST.xlsx")

    dermatology_collumns, df_dermatology = get_demartology_data()
    print("Running Bayes - Dermatology")
    run_test(df_dermatology, dermatology_collumns, method, 15, 17, file="NAIVE_BAYES_DERMATOLOGY.xlsx")


def run_bayes():
    method = BayesClassifier()

    iris_collumns, df_iris = get_iris_data()
    print("Running Bayes - Iris")
    run_test(df_iris, iris_collumns, method, 0, 2, file="BAYES_IRIS.xlsx")

    artificial_collumns, df_artificial = get_artificial_data()
    print("Running Bayes - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, step=0.01, file="BAYES_ARTIFICIAL.xlsx")

    collumn_collumns, df_column = get_collumn_data()
    print("Running Bayes - Collumn")
    run_test(df_column, collumn_collumns, method, 0, 2, step=0.1, file="BAYES_COLLUMN.xlsx")

    breast_collumns, df_breast = get_breast_data()
    print("Running Bayes - Breast")
    run_test(df_breast, breast_collumns, method, 2, 4, file="BAYES_BREAST.xlsx")

    dermatology_collumns, df_dermatology = get_demartology_data()
    print("Running Bayes - Dermatology")
    run_test(df_dermatology, dermatology_collumns, method, 15, 17, file="BAYES_DERMATOLOGY.xlsx")


def run_knn():
    method = Knn()

    iris_collumns, df_iris = get_iris_data()
    print("Running Knn - Iris")
    run_test(df_iris, iris_collumns, method, 0, 2, step=0.1, file="BAYES_IRIS.xlsx")

    artificial_collumns, df_artificial = get_artificial_data()
    print("Running Knn - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, step=0.01, file="BAYES_ARTIFICIAL.xlsx")

    collumn_collumns, df_column = get_collumn_data()
    print("Running Knn - Collumn")
    run_test(df_column, collumn_collumns, method, 0, 2, step=0.1, file="BAYES_COLLUMN.xlsx")

    breast_collumns, df_breast = get_breast_data()
    print("Running Knn - Breast")
    run_test(df_breast, breast_collumns, method, 2, 4, step=0.1, file="BAYES_BREAST.xlsx")

    dermatology_collumns, df_dermatology = get_demartology_data()
    print("Running Knn - Dermatology")
    run_test(df_dermatology, dermatology_collumns, method, 15, 17, step=0.1, file="BAYES_DERMATOLOGY.xlsx")


def run_dmc():
    method = Dmc()

    iris_collumns, df_iris = get_iris_data()
    print("Running DMC - Iris")
    run_test(df_iris, iris_collumns, method, 0, 2, file="BAYES_IRIS.xlsx")

    artificial_collumns, df_artificial = get_artificial_data()
    print("Running DMC - Artificial")
    run_test(df_artificial, artificial_collumns, method, 0, 2, step=0.01, file="BAYES_ARTIFICIAL.xlsx")

    collumn_collumns, df_column = get_collumn_data()
    print("Running DMC - Collumn")
    run_test(df_column, collumn_collumns, method, 0, 2, step=0.1, file="BAYES_COLLUMN.xlsx")

    breast_collumns, df_breast = get_breast_data()
    print("Running DMC - Breast")
    run_test(df_breast, breast_collumns, method, 2, 4, file="BAYES_BREAST.xlsx")

    dermatology_collumns, df_dermatology = get_demartology_data()
    print("Running DMC - Dermatology")
    run_test(df_dermatology, dermatology_collumns, method, 15, 17, file="BAYES_DERMATOLOGY.xlsx")


if __name__ == '__main__':
    run_bayes_discriminant()
    # run_naive_bayes()
    # run_bayes()
    # run_knn()
    # run_dmc()
