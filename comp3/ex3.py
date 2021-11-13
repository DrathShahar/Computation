import numpy as np
from matplotlib import pyplot as plt


# Q1 #

def examples(p):
    x = np.random.uniform(-5, 5, p)
    y = []
    res = []
    for i in range(p):
        y.append(1 + x[i] + x[i] * x[i] + x[i] * x[i] * x[i])
        res.append([x[i], y[i]])

    return res, x, y


def find_perceptron(examples):
    c = find_c(examples)
    u = find_u(examples)
    c_inverse = 1 / c
    w = c_inverse * u
    return w


def find_c(examples):
    sum = 0
    for item in examples:
        sum += item[0] * item[0]
    return sum / len(examples)


def find_u(examples):
    sum = 0
    for item in examples:
        sum += item[0] * item[1]
    return sum / len(examples)


def draw_1_a(w, x, y):
    plt.plot([-5, 5], [w * (-5), w * 5], 'k', label="perceptron results")
    plt.scatter(x, y, label="real results")
    plt.xlabel("x value")
    plt.ylabel("results")
    plt.legend()
    plt.show()


def calc_t_error(w, x, y):
    sum = 0
    for i in range(len(x)):
        diff = w * x[i] - y[i]
        sum += diff * diff
    return 0.5 * (sum / len(x))


def calc_g_error(w):
    sum = 0
    for i in range(-500, 500):
        num = i / 100
        diff = w * num - (1 + num + num ** 2 + num ** 3)
        sum += diff ** 2
    return 0.5 * (sum / 1000)


# Q2 #

def graph_2():
    arr_t = []
    arr_g = []
    p_vals = [i for i in range(5, 105, 5)]
    for p in p_vals:
        t_err = 0
        g_err = 0
        for i in range(100):
            pairs, x, y = examples(p)
            w = find_perceptron(pairs)
            t_err += calc_t_error(w, x, y)
            g_err += calc_g_error(w)
        arr_t.append(t_err/100)
        arr_g.append(g_err/100)
    plt.plot(p_vals, arr_t, 'k', label="t")
    plt.plot(p_vals, arr_g, 'b', label="g")
    plt.xlabel("x value")
    plt.ylabel("error")
    plt.legend()
    plt.show()


# Q3 #
step = 0.01


def gradient_examples(p):
    temp = np.random.uniform(-5, 5, p)
    x = []
    y = []
    res = []
    for i in range(p):
        x.append([temp[i], 1])
        y.append(1 + temp[i] + temp[i] * temp[i] + temp[i] * temp[i] * temp[i])
        res.append([x[i], y[i]])

    return res, x, y


def derivative(pairs, w, idx):
    sum = 0
    other = 1
    if idx == 1:
        other = 0
    for i in range(len(pairs)):
        y = pairs[i][1]
        sum += (w[idx] * pairs[i][0][idx] + w[other] * pairs[i][0][other] - y) * pairs[i][0][idx]
    return sum / len(pairs)


def calc_t_err_q3(pairs, w):
    sum = 0
    for pair in pairs:
        x0 = pair[0][0]
        x1 = pair[0][1]
        y = pair[1]
        diff = w[0] * x0 + w[1] * x1 - y
        sum += diff ** 2
    return 0.5 * sum / len(pairs)


def calc_g_err_q3(w):
    sum = 0
    for i in range(-500, 500):
        num = i / 100
        diff = w[0] * num + w[1] - (1 + num + num ** 2 + num ** 3)
        sum += diff ** 2
    return 0.5 * (sum / 1000)


def gradient_batch(pairs):
    err_t = []
    err_g = []
    w = [1, 1]
    for i in range(100):
        err_t.append(calc_t_err_q3(pairs, w))
        err_g.append(calc_g_err_q3(w))
        w[0] += -step * derivative(pairs, w, 0)
        w[1] += -step * derivative(pairs, w, 1)
    return w, err_t, err_g


def gradient_online(pairs):
    err_t = []
    err_g = []
    w = [1, 1]
    for pair in pairs:
        err_t.append(calc_t_err_q3(pairs, w))
        err_g.append(calc_g_err_q3(w))
        y = pair[1]
        x0 = pair[0][0]
        x1 = pair[0][1]
        w[0] += -step * (w[0] * x0 + w[1] * x1 - y) * x0
        w[1] += -step * (w[0] * x0 + w[1] * x1 - y) * x1
    return w, err_t, err_g


def find_c_q3(pairs):
    c = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            for m in range(len(pairs)):
                c[i][j] += pairs[m][0][i] * pairs[m][0][j]
            c[i][j] /= len(pairs)
    return c


def find_u_q3(pairs):
    u = [0, 0]
    for i in range(2):
        for m in range(len(pairs)):
            x_i = pairs[m][0][i]
            y = pairs[m][1]
            u[i] += x_i * y
        u[i] /= len(pairs)
    return u


def find_inverse(c):
    s = 1/(c[0][0] * c[1][1] - c[1][0] * c[0][1])
    c_inverse = [[0, 0], [0, 0]]
    c_inverse[0][0] = s * c[1][1]
    c_inverse[0][1] = -s * c[1][0]
    c_inverse[1][0] = -s * c[0][1]
    c_inverse[1][1] = s * c[0][0]
    return c_inverse


def matrix_learning(pairs):
    c = find_c_q3(pairs)
    u = find_u_q3(pairs)
    c_inverse = find_inverse(c)
    w = np.array(c_inverse) @ np.array(u)
    err_t = calc_t_err_q3(pairs, w)
    err_g = calc_g_err_q3(w)
    return w, err_t, err_g


pairs, x, y = gradient_examples(100)
w_a, t_a_arr, g_a_arr = gradient_batch(pairs)
w_b, t_b_arr, g_b_arr = gradient_online(pairs)
w_c, t_c, g_c = matrix_learning(pairs)


# Q4 #
def graph_4():
    x = [i + 1 for i in range(100)]
    t_c_arr = [t_c] * 100
    g_c_arr = [g_c] * 100
    plt.plot(x, t_a_arr, label="batch gradient - t")
    plt.plot(x, g_a_arr, label="batch gradient - g")
    plt.plot(x, t_b_arr, label="online gradient - t")
    plt.plot(x, g_b_arr, label="online gradient - g")
    plt.plot(x, t_c_arr, label="matrix learning - t")
    plt.plot(x, g_c_arr, label="matrix learning - g")
    plt.xlabel("number of steps")
    plt.ylabel("error rates")
    plt.legend()
    plt.show()


graph_4()
