import math
import numpy as np
import matplotlib.pyplot as plt


def find_min_error(table_errors, Wr):
    min_threshold = table_errors[0]['threshold']
    min_error = table_errors[0]['error'] + Wr * table_errors[0]['rejection']

    for i in range(len(table_errors)):
        if min_error > (table_errors[i]['error'] + Wr * table_errors[i]['rejection']):
            min_threshold = table_errors[i]['threshold']
            min_error = table_errors[i]['error'] + Wr * table_errors[i]['rejection']

    return min_threshold


def run_test(df, df_collumns, method, startCollumn, stopCollumn, step=0.02, file="result.csv"):
    table_errors = []

    for threshold in np.arange(2, 4, 0.1):
        results = []
        rejections = []
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
            rejection_count = 0
            count = 0

            for n in range(math.floor(len(numpy_array_data_data) * 0.8), len(numpy_array_data_data)):
                predicted_class = method.predict(numpy_array_data_data[n], (threshold+1)/10)

                if predicted_class == -1:
                    rejection_count += 1
                    count += 1
                    continue

                if yLabel[predicted_class] == numpy_array_class_data[n]:
                    correct_count += 1

                count += 1

            results.append(correct_count / (count-rejection_count) if rejection_count < count else 0)
            rejections.append(rejection_count / count)
            datas.append(shuffled_df)

        # Resultados
        print("Resultados: ", results)
        result_data = np.array(results)
        rejections_data = np.array(rejections)

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

        mean = np.mean(result_data)
        rejections_mean = np.mean(rejections_data)
        # Desvio Padrão
        std_dev = np.std(result_data)
        rejections_std_dev = np.std(result_data)

        print("Acurácia (Média):", mean)
        print("Desvio Padrão:", std_dev)

        table_errors.append({
            "threshold": (threshold + 1) / 10,
            "error": 1-mean,
            "Acurácia (Média):": mean,
            "Desvio Padrão:": std_dev,
            "Desvio Padrão (Rejeição):": rejections_std_dev,
            "rejection": rejections_mean
        })

    print(table_errors)
    # Extraindo Acurácia e Rejeição
    acuracia = [item['Acurácia (Média):'] for item in table_errors]
    rejeicao = [item['rejection'] for item in table_errors]

    # Ajustando uma linha reta (polinômio de grau 1)
    coef = np.polyfit(rejeicao, acuracia, 1)
    tendencia = np.polyval(coef, rejeicao)

    # Plotando o gráfico
    plt.plot(rejeicao, acuracia, marker='o')
    plt.plot(rejeicao, tendencia, '--', label='Linha de Tendência')
    plt.title('Curva de Acurácia-Rejeição (AR)')
    plt.xlabel('Rejeição')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.show()

    accurace_rejection = []

    for Wr in [0.04, 0.12, 0.24, 0.36, 0.48]:
        threshold = find_min_error(table_errors, Wr)
        print(threshold, Wr, threshold * Wr)
        results = []
        rejections = []
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
            rejection_count = 0
            count = 0

            for n in range(math.floor(len(numpy_array_data_data) * 0.8), len(numpy_array_data_data)):
                predicted_class = method.predict(numpy_array_data_data[n], Wr * threshold)

                if predicted_class == -1:
                    rejection_count += 1
                    count += 1
                    continue

                if yLabel[predicted_class] == numpy_array_class_data[n]:
                    correct_count += 1

                count += 1

            results.append(correct_count / (count-rejection_count) if rejection_count < count else 0)
            rejections.append(rejection_count / count)
            datas.append(shuffled_df)

        # Resultados
        print("Resultados: ", results)
        result_data = np.array(results)
        rejections_data = np.array(rejections)

        # Média
        mean = np.mean(result_data)
        rejections_mean = np.mean(rejections_data)
        # Desvio Padrão
        std_dev = np.std(result_data)
        rejections_std_dev = np.std(rejections_data)

        print("Acurácia (Média):", mean)
        print("Rejeição (Média):", rejections_mean)
        print("Desvio Padrão:", std_dev)
        print("Desvio Padrão (Rejeição):", rejections_std_dev)

        accurace_rejection.append([mean, rejections_mean])

    # Separando acurácia e rejeição
    acuracia = [item[0] for item in accurace_rejection]
    rejeicao = [item[1] for item in accurace_rejection]

    # Ajustando uma linha reta (polinômio de grau 1)
    print(rejeicao, acuracia)
    # Verificar se os dados são constantes
    if len(set(rejeicao)) == 1 or len(set(acuracia)) == 1:
        print("Os dados são constantes. Não é possível ajustar uma reta.")
    else:
        coef = np.polyfit(rejeicao, acuracia, 1)
        print("Coeficientes da reta:", coef)
    tendencia = np.polyval(coef, rejeicao)

    # Plotando o gráfico
    plt.plot(rejeicao, acuracia, marker='o')
    plt.plot(rejeicao, tendencia, '--', label='Linha de Tendência')
    plt.title('Curva de Acurácia-Rejeição (AR)')
    plt.xlabel('Rejeição')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.show()

