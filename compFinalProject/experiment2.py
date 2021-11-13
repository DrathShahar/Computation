import os
import sys
import time
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import statistics


def experiment(p, q, t, actions_file_name, rewards_file_name):
    actions, rewards = [], []
    for i in range(t):
        action = int(input("Please enter 0 or 1: "))
        while action not in ["0", "1"]:
            action = int(input("Please enter 0 or 1: "))
        reward = np.random.binomial(1, p, 1)[0] if action else np.random.binomial(1, q, 1)[0]
        actions.append(action)
        rewards.append(reward)
        print("Reward: " + str(reward))
        time.sleep(1)
        os.system('cls')

    with open(actions_file_name, 'w') as f:
        for item in actions:
            f.write("%s\n" % item)
    with open(rewards_file_name, 'w') as f:
        for item in rewards:
            f.write("%s\n" % item)


def full_experiment(p, q, t_a, t_b, p_id):
    experiment(p, q, t_a, "first_actions_%s_t_%d.txt" % (p_id, t_a),
               "first_rewards_%s_t_%d.txt" % (p_id, t_a))
    experiment(q, p, t_b, "second_actions_%s_t_%d.txt" % (p_id, t_b),
               "second_rewards_%s_t_%d.txt" % (p_id, t_b))


def display_data(actions, actions2, calc=True, part=''):
    percentages = []
    labels = []
    percentages2 = []
    labels2 = []
    if calc:
        for i in range(len(actions) // BATCH_SIZE):
            percentages.append(
                round(sum(actions[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]) / BATCH_SIZE, 3))
            labels.append(str(i * BATCH_SIZE) + "-" + str((i + 1) * BATCH_SIZE))
            percentages2.append(
                round(sum(actions2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]) / BATCH_SIZE, 3))
            labels2.append(str(i * BATCH_SIZE) + "-" + str((i + 1) * BATCH_SIZE))
    if not calc:
        for i in range(len(actions)):
            labels.append(str(i * BATCH_SIZE) + "-" + str((i + 1) * BATCH_SIZE))
            labels2.append(str(i * BATCH_SIZE) + "-" + str((i + 1) * BATCH_SIZE))
        percentages = actions
        percentages2 = actions2

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, percentages, width, label="REINFORCE")
    rects2 = ax.bar(x+0.25 - width / 2, percentages2, width, label="TD")

    ax.set_ylabel(f"% of Selection=1")
    ax.set_xlabel(f"Selection Batches")
    ax.set_title(f"Selection=1 Percentages According to Batches of \nSelections for {part}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
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
    # plt.savefig(f'{name}.png')
    plt.show()


# PART C - Q1#
def log_likelihood_rein(actions, rewards, eta, w):
    log_sum = 0
    for i, action in enumerate(actions):
        p = 1 / (1 + np.exp(-w))
        p_y_given_eta = p if action else 1 - p
        log_sum += np.log(p_y_given_eta)
        w += eta * rewards[i] * (action - p)
    return log_sum, w


def log_likelihood_td(actions, rewards, eta, v0, v1):
    b = 1
    log_sum = 0
    for i, action in enumerate(actions):
        p = np.exp(b * v1) / (np.exp(b * v1) + np.exp(b * v0))
        p_y_given_eta = p if action else 1 - p
        log_sum += np.log(p_y_given_eta)
        if action:
            v1 += eta * (rewards[i] - v1)
        else:
            v0 += eta * (rewards[i] - v0)
    return log_sum, v0, v1


# PART C - Q2#
def arg_max_likelihood_rein(actions, rewards, w):
    max_eta = 0
    max_val = -np.infty
    final_w = w
    for i in range(200):
        cur_val, temp_w = log_likelihood_rein(actions, rewards, i / 200, w)
        if cur_val > max_val:
            max_eta = i / 200
            max_val = cur_val
            final_w = temp_w
    return max_eta, final_w


def arg_max_likelihood_td(actions, rewards, v0, v1):
    max_eta = 0
    max_val = -np.infty
    final_v0 = v0
    final_v1 = v1
    for i in range(200):
        cur_val, temp_v0, temp_v1 = log_likelihood_td(actions, rewards, i / 200, v0, v1)
        if cur_val > max_val:
            max_eta = i / 200
            max_val = cur_val
            final_v0 = temp_v0
            final_v1 = temp_v1
    return max_eta, final_v0, final_v1


def analyze_participant_data(name):
    first_act_file = open("first_actions_%s.txt" % name, 'r')
    first_actions = [int(line[0]) for line in first_act_file.readlines()]
    first_rewards_file = open("first_rewards_%s.txt" % name, 'r')
    first_rewards = [int(line[0]) for line in first_rewards_file.readlines()]
    second_act_file = open("second_actions_%s.txt" % name, 'r')
    second_actions = [int(line[0]) for line in second_act_file.readlines()]
    second_rewards_file = open("second_rewards_%s.txt" % name, 'r')
    second_rewards = [int(line[0]) for line in second_rewards_file.readlines()]

    res = []
    max_eta, final_w = arg_max_likelihood_rein(first_actions, first_rewards, 0)
    res.append(max_eta)
    max_eta, temp_w = arg_max_likelihood_rein(second_actions, second_rewards, final_w)
    res.append(max_eta)
    max_eta, final_v0, final_v1 = arg_max_likelihood_td(first_actions, first_rewards, 0, 0)
    res.append(max_eta)
    max_eta, temp_v0, temp_v1 = arg_max_likelihood_td(second_actions, second_rewards, final_v0,
                                                      final_v1)
    res.append(max_eta)
    return res


# PART C - Q3#
def simulation_rein(eta1, eta2, p_1, p_0, t1, t2):
    actions = []
    rewards = []
    w = 0
    for i in range(t1):
        p = 1 / (1 + np.exp(-w))
        action = np.random.binomial(1, p, 1)[0]
        if action:
            reward = np.random.binomial(1, p_1, 1)[0]
        else:
            reward = np.random.binomial(1, p_0, 1)[0]
        w += eta1 * reward * (action - p)
        actions.append(action)
        rewards.append(reward)
    for i in range(t2):
        p = 1 / (1 + np.exp(-w))
        action = np.random.binomial(1, p, 1)[0]
        if action:
            reward = np.random.binomial(1, p_0, 1)[0]
        else:
            reward = np.random.binomial(1, p_1, 1)[0]
        w += eta2 * reward * (action - p)
        actions.append(action)
        rewards.append(reward)
    return actions, rewards


def simulation_td(eta1, eta2, p_1, p_0, t1, t2):
    actions = []
    rewards = []
    v0 = 0
    v1 = 0
    b = 1
    for i in range(t1):
        p = np.exp(b * v1) / (np.exp(b * v1) + np.exp(b * v0))
        action = np.random.binomial(1, p, 1)[0]
        if action:
            reward = np.random.binomial(1, p_1, 1)[0]
        else:
            reward = np.random.binomial(1, p_0, 1)[0]
        if action:
            v1 += eta1 * (reward - v1)
        else:
            v0 += eta1 * (reward - v0)
        actions.append(action)
        rewards.append(reward)
    for i in range(t2):
        p = np.exp(b * v1) / (np.exp(b * v1) + np.exp(b * v0))
        action = np.random.binomial(1, p, 1)[0]
        if action:
            reward = np.random.binomial(1, p_0, 1)[0]
        else:
            reward = np.random.binomial(1, p_1, 1)[0]
        if action:
            v1 += eta2 * (reward - v1)
        else:
            v0 += eta2 * (reward - v0)
        actions.append(action)
        rewards.append(reward)
    return actions, rewards


def big_simulation_rein(eta1, eta2, t1, t2):
    etas1 = []
    etas2 = []
    for i in range(100):
        simulation_actions, simulation_rewards = simulation_rein(eta1, eta2, 0.8, 0.4, t1, t2)
        first_actions = simulation_actions[:t1]
        second_actions = simulation_actions[t1:]
        first_rewards = simulation_rewards[:t1]
        second_rewards = simulation_rewards[t1:]

        max_eta, final_w = arg_max_likelihood_rein(first_actions, first_rewards, 0)
        etas1.append(max_eta)
        max_eta, temp_w = arg_max_likelihood_rein(second_actions, second_rewards, final_w)
        etas2.append(max_eta)
    return etas1, etas2


def big_simulation_td(eta1, eta2, t1, t2):
    etas1 = []
    etas2 = []
    for i in range(100):
        simulation_actions, simulation_rewards = simulation_td(eta1, eta2, 0.8, 0.4, t1, t2)
        first_actions = simulation_actions[:t1]
        second_actions = simulation_actions[t1:]
        first_rewards = simulation_rewards[:t1]
        second_rewards = simulation_rewards[t1:]

        max_eta, final_v0, final_v1 = arg_max_likelihood_td(first_actions, first_rewards, 0, 0)
        etas1.append(max_eta)
        max_eta, temp_v0, temp_v1 = arg_max_likelihood_td(second_actions, second_rewards, final_v0,
                                                          final_v1)
        etas2.append(max_eta)
    return etas1, etas2


def histogram(data):
    # bins = np.linspace(-0.1, 1, 100)

    # plt.hist(data, bins, alpha=0.5)
    # plt.legend(loc='upper left')
    # plt.show()
    return sum(data) / len(data), statistics.variance(data)


def simulations_graph(x1, participant_y1, simulations_y1, simulations_e1, x2, participant_y2,
                      simulations_y2, simulations_e2):
    plt.errorbar(x1, simulations_y1, yerr=simulations_e1, fmt='none', capsize=3)
    plt.scatter(x1, participant_y1, color='blue')
    plt.errorbar(x2, simulations_y2, yerr=simulations_e2, fmt='none', ecolor='red', capsize=3)
    plt.scatter(x2, participant_y2, color='red')
    plt.show()


def calculate_models_success(trial_type, etas):
    sims_td = dict()
    sims_r = dict()
    for eta in etas:
        eta = round(eta, 3)
        sims_td[eta], sims_r[eta] = [0] * 160, [0] * 160
        for i in range(100):
            actions_td, rewards_td = simulation_td(eta, eta, 0.8, 0.4, trial_type, 160 - trial_type)
            sims_td[eta] = [sum(x) for x in zip(sims_td[eta], actions_td)]
            actions_r, rewards_r = simulation_rein(eta, eta, 0.8, 0.4, trial_type, 160 - trial_type)
            sims_r[eta] = [sum(x) for x in zip(sims_r[eta], actions_r)]
    for eta in sims_td:
        sims_td[eta] = [round(x / 100, 3) for x in sims_td[eta]]
    for eta in sims_r:
        sims_r[eta] = [round(x / 100, 3) for x in sims_r[eta]]
    return sims_td, sims_r


if __name__ == "__main__":
    BATCH_SIZE = 10
    PROJECT_PATH = 'G:\\My Drive\\University\\Year 3\\Semester A\\06119 - Computation and Cognition\\Final Project'
    ETAS = [0.5]

    for eta in ETAS:
        # avgs_td_60, avgs_r_60 = calculate_models_success(60, [eta])
        avgs_td_80, avgs_r_80 = calculate_models_success(80, [eta])
        # avgs_td_100, avgs_r_100 = calculate_models_success(100, [eta])
        # display_data(f'{PROJECT_PATH}\\Graphs\\avg_60_td_eta_{eta}', avgs_td_60[eta], True,
        #              f'TD Model 60-100 (100 Simulations)')
        # display_data(f'{PROJECT_PATH}\\Graphs\\avg_60_r_eta_{eta}', avgs_r_60[eta], True,
        #              f'REINFORCE Model 60-100 (100 Simulations)')
        # display_data(f'{PROJECT_PATH}\\Graphs\\avg_80_td_eta_{eta}', avgs_td_80[eta], True,
        #              f'TD Model 80-80 (100 Simulations, Eta {eta})')
        # display_data(f'{PROJECT_PATH}\\Graphs\\avg_80_r_eta_{eta}', avgs_r_80[eta], True,
        #              f'REINFORCE Model 80-80 (100 Simulations, Eta {eta})')
        display_data(avgs_r_80[eta],avgs_td_80[eta], True,
                     f'REINFORCE & TDL Models 80-80 (100 Simulations, Eta {eta})')

        # display_data(f'{PROJECT_PATH}\\Graphs\\avg_100_td_eta_{eta}', avgs_td_100[eta], True,
        #              f'TD Model 100-60 (100 Simulations)')
        # display_data(f'{PROJECT_PATH}\\Graphs\\avg_100_r_eta_{eta}', avgs_r_100[eta], True,
        #              f'REINFORCE Model 100-60 (100 Simulations)')
