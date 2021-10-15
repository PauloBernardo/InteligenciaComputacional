from copy import deepcopy
from sat import get_literals
from sat import ALPHABET


def pure_symbol(clausulas, symbol):
    positives = 0
    negatives = 0
    for i in clausulas:
        if 'not({})'.format(symbol) in i:
            negatives += 1
        elif symbol in i:
            positives += 1
    if positives != 0 and negatives != 0:
        return False, None
    elif positives == 0:
        return True, False
    else:
        return True, True


def unit_clausula(clausulas, symbol):
    for cl in clausulas:
        if "not({})".format(symbol) in cl:
            n = 0
            for i in cl:
                if i in ALPHABET:
                    n += 1
            if n == 1:
                return True, False
        elif symbol in cl:
            n = 0
            for i in cl:
                if i in ALPHABET:
                    n += 1
            if n == 1:
                return True, True
    return False, None


def change_clausula(clausulas, model):
    expression = ""
    for i in range(len(clausulas) - 1):
        expression = expression + clausulas[i] + " and "
    expression = expression + clausulas[-1]
    literals = get_literals(expression)
    new_expression = deepcopy(expression)
    for i, v in enumerate(model):
        if literals[i] in new_expression:
            new_expression = new_expression.replace(literals[i], str(v))
    return new_expression


def DPLL(clausulas, symbols, model):
    if len(symbols) == 0:
        return eval(change_clausula(clausulas, model)), model
    _model = deepcopy(model)
    num_clausulas = len(clausulas)
    n = 0
    for i in clausulas:
        sc = str(i)
        try:
            if eval(sc):
                n += 1
            else:
                return False, _model
        except NameError:
            pass
    if n == num_clausulas:
        return True, _model
    first = symbols[0]
    rest = symbols[1:]
    is_pure_symbol, value = pure_symbol(clausulas, first)
    if is_pure_symbol:
        _model.append(value)
        return DPLL(clausulas, symbols[1:], _model)
    is_unit_clausula, value = unit_clausula(clausulas, first)
    if is_unit_clausula:
        _model.append(value)
        return DPLL(clausulas, symbols[1:], _model)
    mtt = deepcopy(_model)
    mtt.append(True)
    rs, new_model = DPLL(clausulas, rest, mtt)
    if rs:
        return rs, new_model
    mtt = deepcopy(_model)
    mtt.append(False)
    rs, new_model = DPLL(clausulas, rest, mtt)
    if rs:
        return rs, new_model
    return False, _model


def DPLLsat(expression):
    clausulas = expression.split("and")
    literals = get_literals(expression)
    return DPLL(clausulas, literals, [])


if __name__ == "__main__":
    # expressions = [
    #     "(not(A) or B)",
    #     "not(C)",
    #     "not(D)",
    #     "E",
    #     "not(X)",
    #     "not(G)",
    #     "not(H)",
    #     "(I or not(J))",
    #     "(K or not(A))",
    #     "(M or N)",
    #     "not(O)",
    #     "not(P)",
    #     "(Q or V)",
    #     "(Q or R)",
    #     "(Q or S)",
    #     "(not(W) or Z or Y)"
    # ]
    # _literals = set([j for i in [get_literals(i) for i in expressions] for j in i])
    # _appreciation = [random.choice([True, False]) for i in range(len(_literals))]
    # print(_literals)
    # print(_appreciation)
    # print(unit_clausula(expressions, 'X'))
    # print(unit_clausula(expressions, 'E'))
    # print(change_clausula(expressions, _appreciation), eval(change_clausula(expressions, _appreciation)))
    # print("---- TESTE -----")
    # exp = " and ".join(expressions)
    # print(exp)
    # rs, appr = DPLLsat(exp)
    # print(eval(change_clausula(expressions, appr)))
    # print(DPLLsat("J and not(R) and (not(L) or R) and (X or D) and (not(I) or V) and H"))
    print(DPLLsat("not(B) and (K or not(G)) and (not(I) or not(E)) and not(A) and not(S) and not(C) and J and (not(E) or D) and not(S) and not(C) and (not(U) or not(R)) and (not(M) or not(O)) and not(N) and (not(U) or D) and not(Q)"))
