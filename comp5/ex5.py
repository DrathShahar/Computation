import numpy as np


def sample(p, f, sigma_x):
    x1 = np.random.uniform(-1, 1, 1)[0]
    option = np.random.binomial(1, p, 1)
    if option:
        eps = np.random.normal(0, sigma_x ** 2, 1)[0]
        x2 = np.sin(f*x1) + eps
    else:
        x2 = np.random.uniform(-1, 1, 1)[0]
    return np.array([x1, x2])


def sample_prototypes(k):
    x1 = np.random.uniform(-1, 1, k)
    x2 = np.random.uniform(-1, 1, k)
    res = []
    for i in range(k):
        res.append([x1[i], x2[i]])
    return res


def init(n, num_of_prototypes, p, f, sigma_x):
    prototypes = sample_prototypes(num_of_prototypes)
    examples = []
    for i in range(n):
        examples.append(sample(p, f, sigma_x))
    return np.array(prototypes), examples


def learning(prototypes, examples, ni, sigma):
    y = np.array([-1] * len(examples))
    for i in range(len(examples)):
        min_dist = dist(examples[i], prototypes[0])
        idx_min = 0

        # find closest prototype:
        for j in range(len(prototypes)):
            cur_dist = dist(examples[i], prototypes[j])
            if cur_dist < min_dist:
                min_dist = cur_dist
                idx_min = j

        y[i] = j
        prototypes = update_prototypes(examples, y, prototypes, ni, sigma, i)
    return prototypes


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def update_prototypes(examples, y, prototypes, ni, sigma, i):
    # delta = [0] * len(prototypes)
    indices = np.array([i for i in range(len(prototypes))])
    yi = [y[i]] * len(prototypes)
    # v = np.array(indices) - np.array(yi)
    # pi_v = np.exp((-1) * (v**2)/(2*(sigma**2)))
    # diff = [0] * len(prototypes)
    # for i in range(len(prototypes)):
    #     diff = dist(np.array(examples[i]), np.array(prototypes[y[i]]))
    # delta = ni * diff * pi_v
    # prototypes += np.array(delta)
    pi = pifunc(sigma, indices, yi)
    delta = ni * (-1 * (prototypes.T - examples)).T * np.array([pi, pi])
    prototypes += delta


def pifunc(sigma, indices, yi):
    a = np.exp(-1 * (1 / (2 * np.square(sigma))) * np.square(indices - yi))
    return a / sum(a)


n = 20000
sigma_x = 0.1
f = 4
p = 0.95
k = 100

prototypes, examples = init(n, k, p, f, sigma_x)

sigma = 4
ni = 1

prototypes = learning(prototypes, examples, ni, sigma)

