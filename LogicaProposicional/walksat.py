import random
from copy import deepcopy

from LogicaProposicional.dpll import change_clausula
from LogicaProposicional.sat import get_literals


def evaluate_clausulas(expression, model):
    literals = get_literals(expression)
    clausulas = expression.split("and")
    for symbol, value in zip(literals, model):
        for i, clausula in enumerate(clausulas):
            clausula = clausula.replace(symbol, str(value))
            clausulas[i] = clausula
    return [eval(clausula) for clausula in clausulas]


def walksat(expression, probability, max_change):
    clausulas = expression.split("and")
    literals = list(set([j for i in [get_literals(i) for i in clausulas] for j in i]))
    literals.sort()

    model = [random.choice([True, False]) for _ in literals]

    for _ in range(max_change):
        clausulas_evaluated = evaluate_clausulas(expression, model)

        if eval(change_clausula(clausulas, model)):
            return True, model, clausulas, clausulas_evaluated

        not_satisfied_clausulas = [i for i, c in enumerate(clausulas_evaluated) if c is False]

        symbols_in_clausula = []
        if len(not_satisfied_clausulas) > 0:
            symbols_in_clausula += get_literals(clausulas[not_satisfied_clausulas[0]])

        if random.randint(1, 100) > probability:
            change_value_symbol = random.choice(list(set(symbols_in_clausula)))
        else:
            change_value_symbol = None
            true_numbers = clausulas_evaluated.count(True)
            for symbol in symbols_in_clausula:
                symbol_position = literals.index(symbol)
                new_model = deepcopy(model)
                new_model[symbol_position] = not new_model[symbol_position]
                new_clausulas_evaluated = evaluate_clausulas(expression, new_model)
                count_true = new_clausulas_evaluated.count(True)
                if count_true > true_numbers:
                    true_numbers = count_true
                    change_value_symbol = symbol
        if change_value_symbol:
            final_position = literals.index(change_value_symbol)
            model[final_position] = not model[final_position]

    return False, model, clausulas, clausulas_evaluated


if __name__ == "__main__":
    print(walksat("(A or B) and (C or D)", 50, 10000))
    print(walksat("(A or B) and (C or D) and not(E) and (X or G)", 50, 10000))
    print(walksat("(A or B) and (C or D) and not(E) and (not(X) or G)", 50, 100000))
    print(walksat("(A or B) and not(C) and not(D) and E and not(X) and (B or D)", 50, 10000))
    print(walksat("not(B) and (K or not(G)) and (not(I) or not(E)) and not(A) and not(S) and not(C) and J and (not(E) or D) and not(S) and not(C) and (not(U) or not(R)) and (not(M) or not(O)) and not(N) and (not(U) or D) and not(Q)", 50, 10000))
