import numpy as np
import matplotlib.pyplot as plt

from utils import loadMNISTLabels, loadMNISTImages
from ff import FF

## Loading the dataset

# y_test = loadMNISTLabels('../MNIST_data/t10k-labels-idx1-ubyte')
y_test = loadMNISTLabels('C:\\Users\\Shahar\\Downloads\\ex4_files\\MNIST_data\\t10k-labels-idx1-ubyte')
# y_train = loadMNISTLabels('../MNIST_data/train-labels-idx1-ubyte')
y_train = loadMNISTLabels('C:\\Users\\Shahar\\Downloads\\ex4_files\\MNIST_data\\train-labels-idx1-ubyte')

# X_test = loadMNISTImages('../MNIST_data/t10k-images-idx3-ubyte')
X_test = loadMNISTImages('C:\\Users\\Shahar\\Downloads\\ex4_files\\MNIST_data\\t10k-images-idx3-ubyte')
# X_train = loadMNISTImages('../MNIST_data/train-images-idx3-ubyte')
X_train = loadMNISTImages('C:\\Users\\Shahar\\Downloads\\ex4_files\\MNIST_data\\train-images-idx3-ubyte')

## random permutation of the input
# uncomment this to use a fixed random permutation of the images

# perm = np.random.permutation(784)
# X_test = X_test[perm,:]
# X_train = X_train[perm,:]

## Parameters
layers_sizes = [784, 30, 10]  # flexible, but should be [784,...,10]
epochs = 10
# epochs = 3
eta = 0.1
batch_size = 20

## Training


net = FF(layers_sizes)
steps, test_acc = net.sgd(X_train, y_train, epochs, eta, batch_size, X_test, y_test)
# print("steps:")
# print(steps)
# print("acc:")
# print(test_acc)

## plotting learning curve and visualizing some examples from test set
plt.scatter(steps[0], test_acc[0])
for i in range(epochs):
    plt.scatter(steps[20*(i+1) - 1]*3, test_acc[20*(i+1) - 1])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# YOUR CODE HERE
