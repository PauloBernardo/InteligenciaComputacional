import random
from pprint import pprint
from LogicaProposicional.dpll import DPLLsat
from LogicaProposicional.sat import ALPHABET, get_literals
import time
import matplotlib.pyplot as plt

from LogicaProposicional.walksat import walksat


def chance(string):
    if random.choice([True, False]):
        return negative(string)
    else:
        return string


def negative(string):
    return 'not(' + string + ')'


def ou(string1, string2):
    return '(' + string1 + ' or ' + string2 + ')'


def generator(m, n):
    output = ''
    flag = False
    for i in range(m):
        if flag:
            flag = False
            pass

        elif random.choices([True, False], weights=[0.8, 0.2])[0] and i != m - 1:
            output += ou(chance(random.choice(ALPHABET[0: n])), chance(random.choice(ALPHABET[0: n]))) + ' and '
            flag = True

        else:
            output += chance(random.choice(ALPHABET[0: n])) + ' and '

    saida = output[:len(output) - 5]
    saida += '\n'
    print(saida)
    return saida


def timer(func, *args):
    start_time = (time.time() * 1000)
    func(*args)
    end_time = (time.time() * 1000)

    return end_time - start_time


if __name__ == "__main__":
    data = []
    count = 1
    while count < 200:
        try:
            exp_gerada = generator(count, 22)
            m = len(exp_gerada.split("and"))
            n = len(get_literals(exp_gerada))
            data.append({
                "expression": exp_gerada,
                "c_for_s": m / n,
                "walksat": timer(walksat, exp_gerada, 50, 1000),
                "dpll": timer(DPLLsat, exp_gerada)
            })
        except SyntaxError:
            pass
        count += 10

    data = sorted(data, key=lambda d: d['c_for_s'])
    pprint(data)

    plt.plot([d['walksat'] for d in data])
    plt.plot([d['dpll'] for d in data])
    plt.xticks(range(len(data)), [round(d['c_for_s'], 1) for d in data], rotation=45)
    plt.legend(['Walksat', 'DPLL'])
    plt.xlabel("Clausulas/Simbolos")
    plt.ylabel("Tempo(ms)")
    plt.show()
