import timeit
from copy import deepcopy

import math

infinity = float(math.inf)


def minimax(actual_board, player=1, alfa=-infinity, beta=infinity):
    a_b = deepcopy(actual_board)
    boards = next_boards(a_b)
    scores = []

    if player == 1:
        for board in boards:
            result = score(board)
            if result == 10 or result == -10 or is_finished(board):
                scores.append(result)
            else:
                sc, st = minimax(board, 1 - player)
                if sc >= beta:
                    return sc, st
                alfa = max(alfa, sc)
                scores.append(sc)

        decided_score = max(scores)
    else:
        for board in boards:
            result = score(board)
            if result == 10 or result == -10 or is_finished(board):
                scores.append(result)
            else:
                sc, st = minimax(board, 1 - player)
                if sc <= alfa:
                    return sc, st
                beta = min(beta, sc)
                scores.append(sc)

        decided_score = min(scores)
    decided_index = scores.index(decided_score)
    return decided_score, boards[decided_index]


def finish(board):
    result = score(board)
    if result == 10:
        print("Game over, max won")
        return True
    if result == -10:
        print("Game over, min won")
        return True
    if is_finished(board):
        print("Draw, no one won")
        return True

    return False


def next_boards(actual_board):
    boards = []
    num_1_counter = 0
    num_0_counter = 0
    for _ in actual_board:
        for i in _:
            if i == 1:
                num_1_counter += 1
            elif i == 0:
                num_0_counter += 1

    if num_1_counter > num_0_counter:
        new_value = 0
    else:
        new_value = 1

    for i, j in fields_available(actual_board):
        new = deepcopy(actual_board)
        new[i][j] = new_value
        boards.append(new)

    return boards


def fields_available(board):
    fields = []
    for line in range(len(board)):
        for column in range(len(board[line])):
            if board[line][column] is None:
                fields.append((line, column))
    return fields


def score(board):
    rl_1 = 0
    rl_0 = 0
    lr_1 = 0
    lr_0 = 0
    for i, line in enumerate(board):
        if line.count(1) == 3:
            return 10
        if line.count(0) == 3:
            return -10
        v_1 = 0
        v_0 = 0
        for column in board:
            if column[i] == 1:
                v_1 += 1
            elif column[i] == 0:
                v_0 += 1
        if v_1 == 3:
            return 10
        if v_0 == 3:
            return -10

        # right to left
        if board[i][i] == 1:
            rl_1 += 1
        if board[i][i] == 0:
            rl_0 += 1
        if rl_1 == 3:
            return 10
        if rl_0 == 3:
            return -10

        # left to right
        if board[i][2 - i] == 1:
            lr_1 += 1
        if board[i][2 - i] == 0:
            lr_0 += 1
        if lr_1 == 3:
            return 10
        if lr_0 == 3:
            return -10

    return 0


def print_board(board):
    print('Board:')
    for _ in board:
        for f in _:
            value = '__'
            if f == 1:
                value = 'X'
            elif f == 0:
                value = 'O'
            print(value, end='\t')
        print()


def create_board(tam):
    return [[None for _ in range(tam)] for _ in range(tam)]


def is_finished(board):
    for _ in board:
        for f in _:
            if f is None:
                return False
    return True


def run():
    board = create_board(3)
    print("Loading...")
    while True:
        start = timeit.default_timer()
        player = minimax(board)
        fim = timeit.default_timer()
        print_board(player[1])
        print('Duration of the move: %f sec' % (fim - start))
        if finish(player[1]):
            return
        while True:
            k = eval(input("Choose your move (0...8): "))
            a, b = int(k/3), k % 3
            if 0 <= k <= 8 and player[1][a][b] is None:
                break
            print("Wrong movement! Try again!")
        player[1][a][b] = 0
        print_board(player[1])
        if finish(player[1]):
            return
        board = player[1]


if __name__ == "__main__":
    run()
