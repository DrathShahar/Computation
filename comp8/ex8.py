import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt


# PART A #
def experiment(p, q, t, p_id):
    actions = []
    rewards = []
    for i in range(t):
        action = input(str(i) + " Please enter 0 or 1: ")
        while action not in ["0", "1"]:
            action = input("Please enter 0 or 1: ")
        action = int(action)
        if action:
            reward = np.random.binomial(1, p, 1)[0]
        else:
            reward = np.random.binomial(1, q, 1)[0]
        actions.append(action)
        rewards.append(reward)
        print("reward: " + str(reward))
        time.sleep(1)
        os.system('cls')

    file_name = "actions_%s.txt" % p_id
    with open(file_name, 'w') as f:
        for item in actions:
            f.write("%s\n" % item)
    file_name = "rewards_%s.txt" % p_id
    with open(file_name, 'w') as f:
        for item in rewards:
            f.write("%s\n" % item)


# PART B #
def display_data():
    p_start_1 = sum(actions_1[0:20])/20
    p_end_1 = sum(actions_1[80:])/20
    p_start_2 = sum(actions_2[0:20])/20
    p_end_2 = sum(actions_2[80:])/20

    labels = ['firat 20 trials', 'last 20 trials']
    p_1 = [p_start_1, p_end_1]
    p_2 = [p_start_2, p_end_2]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, p_1, width, label='Participant 1')
    rects2 = ax.bar(x + width / 2, p_2, width, label='Participant 2')

    ax.set_ylabel('Percentage of select 1')
    ax.set_title('Correct Selection (=1) Percentages in the first/last 20 trials')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()


# PART C - Q1#
def log_likelihood_rein(actions, rewards, eta):
    w = 0
    log_sum = 0
    for i, action in enumerate(actions):
        p = 1 / (1 + np.exp(-w))
        if action:
            p_y_given_eta = p
        else:
            p_y_given_eta = 1 - p
        log_sum += np.log(p_y_given_eta)
        w += eta * rewards[i] * (action - p)

    return log_sum


def log_likelihood_td(actions, rewards, eta):
    b = 1
    v0 = 0
    v1 = 0
    log_sum = 0
    for i, action in enumerate(actions):
        p = np.exp(b * v1) / (np.exp(b * v1) + np.exp(b * v0))
        if action:
            p_y_given_eta = p
        else:
            p_y_given_eta = 1 - p
        log_sum += np.log(p_y_given_eta)
        if action:
            v1 += eta * (rewards[i] - v1)
        else:
            v0 += eta * (rewards[i] - v0)

    return log_sum


# PART C - Q2#

def arg_max_likelihood_rein(actions, rewards):
    step = 0.05
    max_eta = 0
    max_val = -np.infty
    for i in range(20):
        cur_val = log_likelihood_rein(actions, rewards, i/20)
        if cur_val > max_val:
            max_eta = i/20
            max_val = cur_val
    return max_eta


def arg_max_likelihood_td(actions, rewards):
    step = 0.05
    max_eta = 0
    max_val = -np.infty
    for i in range(20):
        cur_val = log_likelihood_td(actions, rewards, i/20)
        if cur_val > max_val:
            max_eta = i/20
            max_val = cur_val
    return max_eta


# PART C - Q3#
def simulation_rein(eta, p_1, p_0):
    actions = []
    rewards = []
    w = 0
    for i in range(100):
        p = 1 / (1 + np.exp(-w))
        action = np.random.binomial(1, p, 1)[0]
        if action:
            reward = np.random.binomial(1, p_1, 1)[0]
        else:
            reward = np.random.binomial(1, p_0, 1)[0]
        w += eta * reward * (action - p)
        actions.append(action)
        rewards.append(reward)
    return actions, rewards


def simulation_td(eta, p_1, p_0):
    actions = []
    rewards = []
    v0 = 0
    v1 = 0
    b = 1
    for i in range(100):
        p = np.exp(b * v1) / (np.exp(b * v1) + np.exp(b * v0))
        action = np.random.binomial(1, p, 1)[0]
        if action:
            reward = np.random.binomial(1, p_1, 1)[0]
        else:
            reward = np.random.binomial(1, p_0, 1)[0]
        if action:
            v1 += eta * (reward - v1)
        else:
            v0 += eta * (reward - v0)
        actions.append(action)
        rewards.append(reward)
    return actions, rewards


def big_simulation_rein(eta):
    etas = []
    for i in range(100):
        simulation_actions, simulation_rewards = simulation_rein(eta, 0.8, 0.4)
        etas.append(arg_max_likelihood_rein(simulation_actions, simulation_rewards))
    return etas


def big_simulation_td(eta):
    etas = []
    for i in range(100):
        simulation_actions, simulation_rewards = simulation_td(eta, 0.8, 0.4)
        etas.append(arg_max_likelihood_td(simulation_actions, simulation_rewards))
    return etas


def histogram(participant_1, participant_2):
    x = participant_1
    y = participant_2
    print("average 1: " + str(sum(x)/len(x)))
    print("average 2: " + str(sum(y)/len(y)))

    bins = np.linspace(-0.1, 1, 100)

    plt.hist(x, bins, alpha=0.5, label='participant 1')
    plt.hist(y, bins, alpha=0.5, label='participant 2')
    plt.legend(loc='upper left')
    plt.show()




p_0 = 0.4
p_1 = 0.8
trials = 100
participant_id = sys.argv[1]
experiment(p_1, p_0, trials, participant_id)

# with open("actions_1.txt") as f:
#     content = f.readlines()
# actions_1 = [int(x.strip()) for x in content]
# with open("rewards_1.txt") as f:
#     content = f.readlines()
# rewards_1 = [int(x.strip()) for x in content]
# with open("actions_2.txt") as f:
#     content = f.readlines()
# actions_2 = [int(x.strip()) for x in content]
# with open("rewards_2.txt") as f:
#     content = f.readlines()
# rewards_2 = [int(x.strip()) for x in content]
#
# # display_data()
# max_eta_1_rein = arg_max_likelihood_rein(actions_1, rewards_1)
# max_eta_2_rein = arg_max_likelihood_rein(actions_2, rewards_2)
# max_eta_1_td = arg_max_likelihood_td(actions_1, rewards_1)
# max_eta_2_td = arg_max_likelihood_td(actions_2, rewards_2)
# print("participant 1, rein: " + str(max_eta_1_rein))
# print("participant 2, rein: " + str(max_eta_2_rein))
# print("participant 1, td: " + str(max_eta_1_td))
# print("participant 2, td: " + str(max_eta_2_td))
# histogram(big_simulation_rein(max_eta_1_rein), big_simulation_rein(max_eta_2_rein))
# histogram(big_simulation_td(max_eta_1_td), big_simulation_td(max_eta_2_td))
