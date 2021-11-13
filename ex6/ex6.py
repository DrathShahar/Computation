import scipy.io
import numpy as np
from matplotlib import pyplot as plt


# Q1
def load_data(file_name):
    file = scipy.io.loadmat(file_name)
    data = file["data"]
    labels = file["labels"]
    test_data = file["test_data"]
    test_labels = file["test_labels"]
    return data, labels, test_data, test_labels


# Q2
def learning(examples, correct):
    w = np.random.normal(0, sigma, n)
    accuracy_rates = []
    for i, example in enumerate(examples):
        p = 1/(1+np.exp(-w @ example.T))
        y = np.random.binomial(1, p, 1)[0]
        r = 1 if (correct[i] == y) else 0
        net_sum = w @ example.T
        e_func = (y-1) * example + ((example*np.exp(-1 * net_sum))/(1+np.exp(-1 * net_sum)))
        delta_w = eta * r * e_func
        w += delta_w
        if i % step == 0:
            accuracy_rates.append(calc_accuracy(test_data.T, test_labels[0], w))
    return w, accuracy_rates


# Q3
def calc_accuracy(examples, correct, w):
    counter = 0
    for i, example in enumerate(examples):
        p = 1 / (1 + np.exp(-w @ example.T))
        y = np.random.binomial(1, p, 1)[0]
        r = 1 if (correct[i] == y) else 0
        counter += r
    return counter / len(examples)


# Q4
def display_accuracy_rates(values):
    num_of_examples = [step*i for i in range(len(values))]
    plt.plot(num_of_examples, values)
    plt.title("Accuracy Rates as Function of Number of Examples")
    plt.xlabel("number of examples")
    plt.ylabel("accuracy rate")
    plt.show()


# Q5
def display_w(w):
    h = 28
    plt.imshow(np.reshape(w, (h, h)))
    plt.colorbar(pad=0.2).ax.set_title("color-value map")
    plt.title("w Image")
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.show()


file_name = "ex6_data.mat"
data, labels, test_data, test_labels = load_data(file_name)
eta = 0.01
sigma = 0.01
n = 784
step = 50
w, success_rates = learning(data.T, labels[0])
display_accuracy_rates(success_rates)
display_w(w)
