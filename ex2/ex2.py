import numpy as np
from matplotlib import pyplot as plt


def find_perceptron(x, y):  # Q2.1
    n = len(y)
    counter = 0
    w = np.array([1]*len(x[0]))
    while counter < n:
        counter = 0
        for i in range(n):
            res = w @ np.array(x[i])
            if res * y[i] > 0:
                counter = counter + 1
            else:
                w = w + (y[i] * x[i])
    return w


def init(n):  # Q2.2
    x = np.random.uniform(low=-10.0, high=10.0, size=(n,2))
    y = [0] * n
    for i in range(len(x)):
        if x[i, 0] > x[i, 1]:
            # color = 'blue'
            y[i] = 1
        else:
            # color = 'red'
            y[i] = -1
        # plt.scatter(x[i, 0], x[i, 1], color=color)
    # plt.show()
    return x, y


def classify(x, y):  # Q2.3
    w = find_perceptron(x, y) / 10
    for i in range(len(x)):
        if w @ np.array(x[i]) > 0:
            color = 'blue'
            y[i] = 1
        else:
            color = 'red'
            y[i] = -1
        plt.scatter(x[i, 0], x[i, 1], color=color)
    plt.plot([0, w[0]], [0, w[1]], 'k')
    plt.scatter(w[0], w[1], color='black', marker="^")
    x_values = [w[1]*1.5, w[1]*(-1.5)]
    y_values = [w[0]*(-1.5), w[0]*1.5]
    plt.plot(x_values, y_values, 'm')
    plt.show()


def error_rate():  # Q2.4
    opt = np.array([1, -1])
    p_vals = [5, 20, 30, 50, 100, 150, 200, 500]
    m = 100
    error_vals = []
    for p in p_vals:
        sum = 0
        for i in range(100):
            x, y = init(p)
            w = find_perceptron(x, y)
            cos_alpha = (w @ opt) / (np.linalg.norm(w) * np.linalg.norm(opt))
            sum += abs(np.rad2deg(np.arccos(cos_alpha)))
        error_vals.append(sum / 100)
    return error_vals


x_values = [5, 20, 30, 50, 100, 150, 200, 500]
y_values = error_rate()
plt.plot(x_values, y_values, 'm')
plt.scatter(x_values, y_values, color='black')
plt.title("Average error rate as a function of number of samples")
plt.xlabel("Number of Samples")
plt.ylabel("Average Error Rate [degrees]")
plt.show()
