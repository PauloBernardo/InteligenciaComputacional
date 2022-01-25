from copy import copy
from pprint import pprint

import math


def calculate_entropia(values):
    c_true = values.count(True) / len(values)
    c_false = values.count(False) / len(values)
    if not c_true or not c_false:
        return 0
    return -c_true * math.log(c_true, 2) - c_false * math.log(c_false, 2)


def decision_tree(dataset, level=0):
    copy_dataset = copy(dataset)
    if len(dataset[0]) < 3:
        print(''.join(['\t' for _ in range(level+1)]), 'Result:', 'Not found!')
        return
    dataset = list(map(list, zip(*dataset)))
    I = calculate_entropia(dataset[-1][1:])

    if I == 0:
        print(''.join(['\t' for _ in range(level+1)]), 'Result:', dataset[-1][1])
        return
    attributes = dataset[1:-1]

    average_i = []
    for attribute in attributes:
        visited = []
        Im = 0
        for attribute_value in attribute[1:]:
            values = []
            if attribute_value not in visited:
                for j, attribute_value_aux in enumerate(attribute[1:]):
                    if attribute_value == attribute_value_aux:
                        values.append(dataset[-1][j+1])
                visited.append(attribute_value)
                Im += attribute.count(attribute_value)/len(attribute[1:]) * calculate_entropia(values)
        average_i.append(Im)
    max_i = 0
    chose = 0
    for i, Im in enumerate(average_i):
        if max_i < I - Im:
            max_i = I - Im
            chose = i
    print(''.join(['\t' for _ in range(level)]), dataset[chose+1][0], level)
    visited = []
    for attribute_value in dataset[chose+1][1:]:
        if attribute_value not in visited:
            print(''.join(['\t' for _ in range(level)]), '-->', attribute_value)
            filtered_dataset = [[]]
            count = 0
            for i, line in enumerate(copy_dataset):
                if i == 0 or attribute_value in line:
                    for j, collumn in enumerate(line):
                        if j != chose+1:
                            filtered_dataset[count].append(collumn)
                    filtered_dataset.append([])
                    count += 1
            filtered_dataset = filtered_dataset[0: -1]
            visited.append(attribute_value)
            decision_tree(filtered_dataset, level+1)

    return


if __name__ == '__main__':
    dataset_1 = [
        ['N', 'Clientela', 'Tipo', 'Ficou?'],
        [1, 'lotado', 'à la carte', False],
        [2, 'médio', 'à la carte', True],
        [3, 'vazio', 'à la carte', False],
        [4, 'lotado', 'self service', False],
        [5, 'médio', 'self service', True],
        [6, 'vazio', 'self service', False],
        [7, 'lotado', 'fast food', False],
        [8, 'médio', 'fast food', True],
        [9, 'vazio', 'fast food', False],
        [10, 'lotado', 'sushi bar', False],
        [11, 'médio', 'sushi bar', True],
        [12, 'vazio', 'sushi bar', False],
    ]
    dataset_2 = [
        ['N', 'Tempo', 'Temperatura', 'Humidade', 'Ventando', 'Classe = Foi jogar tenis?'],
        [1, 'Ensolarado', 'Quente', 'Alta', 'Sim', False],
        [2, 'Ensolarado', 'Quente', 'Alta', 'Sim', False],
        [3, 'Nublado', 'Quente', 'Alta', 'Não', True],
        [4, 'Chuva', 'Média', 'Alta', 'Não', True],
        [5, 'Chuva', 'Fria', 'Normal', 'Não', True],
        [6, 'Chuva', 'Fria', 'Normal', 'Sim', False],
        [7, 'Nublado', 'Fria', 'Normal', 'Sim', True],
        [8, 'Ensolarado', 'Média', 'Alta', 'Não', False],
        [9, 'Ensolarado', 'Fria', 'Normal', 'Não', True],
        [10, 'Chuva', 'Média', 'Normal', 'Não', True],
        [11, 'Ensolarado', 'Média', 'Normal', 'Sim', True],
        [12, 'Nublado', 'Média', 'Alta', 'Sim', True],
        [13, 'Nublado', 'Quente', 'Normal', 'Não', True],
        [14, 'Chuva', 'Média', 'Alta', 'Sim', False],
    ]
    dataset_3 = [
        ['Student', 'Prior Experience', 'Course', 'Time', 'Liked'],
        [1, True, 'Programming', 'Day', True],
        [2, False, 'Programming', 'Day', False],
        [3, True, 'History', 'Night', False],
        [4, False, 'Programming', 'Night', True],
        [5, True, 'English', 'Day', True],
        [6, False, 'Programming', 'Day', False],
        [7, True, 'Programming', 'Day', False],
        [8, True, 'Mathematics', 'Night', True],
        [9, True, 'Programming', 'Night', True],
        [10, True, 'Programming', 'Night', False],
    ]

    print("Hello World! I'm going to do a decision tree!")
    print("Test 1:")
    print("Dataset: ")
    pprint(dataset_1)
    print("Decision tree:")
    decision_tree(dataset_1)
    print("---------------------||-------------------")
    print("Test 2:")
    print("Dataset: ")
    pprint(dataset_2)
    print("Decision tree:")
    decision_tree(dataset_2)
    print("---------------------||-------------------")
    print("Test 3:")
    print("Dataset: ")
    pprint(dataset_3)
    print("Decision tree:")
    decision_tree(dataset_3)
