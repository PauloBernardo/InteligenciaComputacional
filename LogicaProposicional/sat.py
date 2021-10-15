ALPHABET = "ABCDEGHIJKLMNOPQRSUVWXYZ"


def get_literals(expression):
    # print(expression)
    atoms = []
    exp_split = expression.split()
    for i in exp_split:
        # print(i)
        i = i.replace("(", "")
        i = i.replace(")", "")
        i = i.replace("not", "")
        if i in ALPHABET and i not in atoms:
            # print('_', i)
            atoms.append(i)
    return atoms


def get_appreciation(tam):
    if tam == 1:
        return [[True], [False]]
    res = []
    for i in get_appreciation(tam-1):
        res.append([True] + i)
    for i in get_appreciation(tam-1):
        res.append([False] + i)
    return res


def sat(expression):
    literals = get_literals(expression)
    print(literals)
    appreciation = get_appreciation(len(literals))
    for i in appreciation:
        new_appreciation = expression
        for l, v in zip(literals, i):
            new_appreciation = new_appreciation.replace(l, str(v))
        if eval(new_appreciation):
            return literals, i
    return False


if __name__ == "__main__":
    print(sat("not(A or B) and (C or D)"))
