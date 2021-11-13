import copy
import math
import numpy as np


# Q2.1
def init_matrix(n, p, f):
    patterns = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            patterns[i, j] = np.random.binomial(1, f, 1)

    copy_patt = copy.deepcopy(patterns)
    J = np.zeros((n, n))
    k = 1 / (f * (1 - f) * n)
    for i in range(n):
        for j in range(i + 1, n):
            sigma = (copy.deepcopy(copy_patt[i]) - f) @ (copy.deepcopy(copy_patt[j]) - f)
            J[i, j] = k * sigma
            J[j, i] = k * sigma

    return patterns, J


# Q2.2
def dynamics(J, s, t):
    n = len(s)
    counter = 0  # counter helps to know if dynamics stops
    while counter < n:
        counter = 0
        for i in range(n):
            if h(i, J, s) > t:
                value = 1
            else:
                value = 0
            if value == s[i]:
                counter = counter + 1
            s[i] = value
    return s


# Q2.2 helper
def h(i, J, s):
    return J[i] @ s


# Q2.3
def stability(J, s, T):
    s_copy = copy.deepcopy(s)
    s_copy = dynamics(J, s_copy, T)
    diff = np.sum(np.abs(np.subtract(s, s_copy)))
    return float(diff) / float(len(s))


# Q2.4
def tests(f):
    N = 100
    a = 0.02
    d = 0.04
    t = 0.5 - f
    results = []
    while a <= 0.8:
        P = math.ceil(N * a)
        patt, J = init_matrix(N, P, f)
        sum_of_stab = 0
        for i in range(5):
            sum_of_stab = sum_of_stab + stability(J, patt[:,0], t)
        results.append(sum_of_stab / 5.0)
        a = a + d
    return results


print(tests(0.1))
print(tests(0.2))
print(tests(0.3))
