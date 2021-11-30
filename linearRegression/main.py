import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def simple_example():
    X = [10, 20, 30]
    Y = [15, 19, 45]
    plt.scatter(X, Y,)
    plt.show()

    A = np.array([10, 1, 20, 1, 30, 1]).reshape(3, 2)
    B = np.array(Y).reshape(3, 1)

    a = np.linspace(10, 30)
    arr = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), Y)
    arr.tolist()
    beta, alpha = arr
    Yi = alpha + beta * a

    plt.scatter(X, Y)
    plt.plot(a, Yi)
    plt.show()


def linear_least_squares(examples):
    m, n = examples.shape
    cx = examples[0].reshape(n, 1)
    c2 = np.ones(len(cx)).reshape(n, 1)
    A = np.hstack((cx, c2))

    return np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), examples[1])


def plot_figure(x, y, alpha, beta, title, x_label, y_label):
    min_y = alpha + beta * min(x)
    max_y = alpha + beta * max(x)

    plt.plot([min(x), max(x)], [min_y, max_y])
    plt.scatter(x, y, color='orange')
    plt.xlabel(x_label)
    plt.title(title)
    plt.grid(True)
    plt.ylabel(y_label)
    plt.show()


def plot_linear_regression(examples, title='Linear Least Squares Regression Example', x_label='X', y_label='Y'):
    min_x = min(examples[0])
    max_x = max(examples[0])
    theta = linear_least_squares(examples)
    theta.tolist()
    beta, alpha = theta

    min_y = alpha + beta * min_x
    max_y = alpha + beta * max_x

    plt.plot([min(examples[0]), max(examples[0])], [min_y, max_y])
    plt.scatter(examples[0], examples[1], color='orange')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()


def simple_linear_least_squares_example():
    plot_linear_regression(np.array([[1.5, 1.6, 1.3, 1.4, 1.5, 1.7, 1.8, 1.7, 1.1, 1.2], [10, 12, 16, 13, 15, 11, 8, 10, 18, 13]]), x_label='Prices', y_label='Sales')


def statistic_linear_regression(x, y):
    number_of_elements = len(x)
    if number_of_elements != len(y):
        raise Exception("Size of x and y must be equal!")
    mean_x, mean_y = sum(x)/number_of_elements, sum(y)/number_of_elements
    sum_x_vezes_y = sum([i * j for i, j in zip(x, y)])
    sum_x_pow_2 = sum([i ** 2 for i in x])
    sxy = sum_x_vezes_y - number_of_elements * mean_x * mean_y
    sxx = sum_x_pow_2 - number_of_elements * mean_x * mean_x
    beta = sxy / sxx
    alpha = mean_y - beta * mean_x
    return alpha, beta


def plot_statistic_linear_regression(x, y, title='Statistic Linear Regression Example', x_label='X', y_label='Y'):
    alpha, beta = statistic_linear_regression(x, y)
    plot_figure(x, y, alpha, beta, title, x_label, y_label)


def simple_statistic_linear_regression_example():
    plot_statistic_linear_regression([1.5, 1.6, 1.3, 1.4, 1.5, 1.7, 1.8, 1.7, 1.1, 1.2], [10, 12, 16, 13, 15, 11, 8, 10, 18, 13], x_label='Prices', y_label='Sales')


def sklearn_linear_regression(x, y):
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    return reg.intercept_[0], reg.coef_[0][0]


def plot_sklearn_linear_regression(x, y, title='Sklearn Linear Regression Example', x_label='X', y_label='Y'):
    alpha, beta = sklearn_linear_regression(x, y)
    plot_figure(x, y, alpha, beta, title, x_label, y_label)


def simple_sklearn_linear_regression_example():
    prices = np.array([1.5, 1.6, 1.3, 1.4, 1.5, 1.7, 1.8, 1.7, 1.1, 1.2])
    sales = np.array([10, 12, 16, 13, 15, 11, 8, 10, 18, 13])
    plot_sklearn_linear_regression(prices, sales, x_label='Prices', y_label='Sales')


if __name__ == '__main__':
    simple_linear_least_squares_example()
    simple_statistic_linear_regression_example()
    simple_sklearn_linear_regression_example()
