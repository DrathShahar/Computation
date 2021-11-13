import numpy as np
from matplotlib import pyplot as plt


# PART A - Q4 #
def value_iteration(v):
    epsilon = 0.000000000000001
    t_max = 5000
    num_of_iterations = 0
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(v):
            num_of_iterations += 1
            sum_stay = calc_sum_stay(v, i)
            sum_switch = calc_sum_switch(v, i)
            stay_v = r[i][0] + gamma * sum_stay
            switch_v = r[i][1] + gamma * sum_switch
            new_v = max(stay_v, switch_v)
            if abs(new_v - v[i]) > epsilon or num_of_iterations < t_max:
                changed = True
            v[i] = new_v
    return v


def calc_sum_stay(v, s_i):
    cur_sum = 0
    for i, s in enumerate(v):
        cur_sum += p[s_i][0][i] * v[i]
    return cur_sum


def calc_sum_switch(v, s_i):
    cur_sum = 0
    for i, s in enumerate(v):
        cur_sum += p[s_i][1][i] * v[i]
    return cur_sum


# PART B - Q1 #

def next_state(s, a):
    probabilities = p[s][a]
    next_s = np.random.binomial(1, probabilities[1], 1)[0]
    return next_s, r[s][a]


# PART B - Q2 #

def learning_v_pi_td(v):
    p = 0.5
    eta = 0.01
    t = 3000
    s = np.random.binomial(1, p, 1)[0]
    v_home = []
    v_out = []
    for i in range(t):
        a = np.random.binomial(1, p, 1)[0]
        new_s, r = next_state(s, a)
        v[s] = v[s] + eta * (r + gamma * v[new_s] - v[s])
        v_home.append(v[0])
        v_out.append(v[1])
        s = new_s
    num_of_steps = [i for i in range(t)]
    plt.plot(num_of_steps, v_out, label="Out")
    plt.plot(num_of_steps, v_home, label="Home")
    y_out = 33/19
    y_home = 23/19
    plt.axhline(y=y_out, color='g', linestyle='--', label="Out Value - Q1")
    plt.axhline(y=y_home, color='r', linestyle='--', label="Home Value - Q1")
    plt.title("Values of States as Function of Number of Steps")
    plt.xlabel("Number of Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# PART B - Q3 #

def learning_v_pi_q(q):
    p = 0.5
    eta = 0.1
    t = 3000
    s = np.random.binomial(1, p, 1)[0]
    v_opt_home = []
    v_opt_out = []
    for i in range(t):
        a = np.random.binomial(1, p, 1)[0]
        new_s, r = next_state(s, a)
        max_q = max(q[new_s][0], q[new_s][1])
        q[s][a] = q[s][a] + eta * (r + gamma * max_q - q[s][a])
        v_opt_home.append(max(q[0][0], q[0][1]))
        v_opt_out.append(max(q[1][0], q[1][1]))
        s = new_s
    num_of_steps = [i for i in range(t)]
    plt.plot(num_of_steps, v_opt_out, label="Out")
    plt.plot(num_of_steps, v_opt_home, label="Home")
    y_out = 4
    y_home = 26/9
    plt.axhline(y=y_out, color='g', linestyle='--', label="Out Value - Q4")
    plt.axhline(y=y_home, color='r', linestyle='--', label="Home Value - Q4")
    plt.title("Values of States as Function of Number of Steps")
    plt.xlabel("Number of Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


gamma = 0.5
default_v = [0, 0]
r = [[0, 1], [2, 0]]
p = [[[1, 0], [0.2, 0.8]], [[0, 1], [1, 0]]]
v = value_iteration(default_v)
print(v)
default_v = [0, 0]
learning_v_pi_td(default_v)
default_q = [[0, 0], [0, 0]]
learning_v_pi_q(default_q)