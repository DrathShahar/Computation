import os
import sys
import time
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import statistics

MAX_ETA = 1
MIN_ETA = 0
STEP_ETA = 0.005
NAMES = ["roby", "miriam", "eyal", "hagar", "tamar", "ran", "ilan", "gil", "amos"]
PARTICIPANTS_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SHIFTED_PARTICIPANTS_NUMBERS = [1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3]
T1 = [80, 80, 100, 80, 80, 60, 100, 100, 60]
T2 = [80, 80, 60, 80, 80, 100, 60, 60, 100]
NUMBER_OF_SIMULATIONS = 100
P_0 = 0.4
P_1 = 0.8
DEFAULT_PART_LEN = 80


def experiment(p, q, t, actions_file_name, rewards_file_name):
    actions = []
    rewards = []
    for i in range(t):
        action = input("Please enter 0 or 1: ")
        while action not in ["0", "1"]:
            action = input("Please enter 0 or 1: ")
        action = int(action)
        if action:  # action == 1
            reward = np.random.binomial(1, p, 1)[0]
        else:
            reward = np.random.binomial(1, q, 1)[0]
        actions.append(action)
        rewards.append(reward)
        print("reward: " + str(reward))
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


def display_data(actions):
    percentages = []
    labels = []
    for i in range(len(actions)//20):
        percentages.append(sum(actions[i*20:(i+1)*20])/20)
        labels.append("trials " + str(i*20)+"-"+str((i+1)*20))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, percentages, width)

    ax.set_ylabel('Percentage of select 1')
    ax.set_title('title')
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
    fig.tight_layout()
    plt.show()


# PART C - Q1#
def log_likelihood_rein(actions, rewards, eta, w):
    log_sum = 0
    for i, action in enumerate(actions):
        p = 1 / (1 + np.exp(-w))
        if action:
            p_y_given_eta = p
        else:
            p_y_given_eta = 1 - p
        log_sum += np.log(p_y_given_eta)
        w += eta * rewards[i] * (action - p)
    return log_sum, w


def log_likelihood_td(actions, rewards, eta, v0, v1):
    b = 1
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
    return log_sum, v0, v1


# PART C - Q2#
def arg_max_likelihood_rein(actions, rewards, w):
    max_eta = 0
    max_val = -np.infty
    final_w = w
    likelihood = []
    for eta in np.arange(MIN_ETA, MAX_ETA, STEP_ETA):
        cur_val, temp_w = log_likelihood_rein(actions, rewards, eta, w)
        likelihood.append(cur_val)
        if cur_val > max_val:
            max_eta = eta
            max_val = cur_val
            final_w = temp_w
    return max_eta, final_w


def arg_max_likelihood_td(actions, rewards, v0, v1):
    max_eta = 0
    max_val = -np.infty
    final_v0 = v0
    final_v1 = v1
    likelihood = []
    for eta in np.arange(MIN_ETA, MAX_ETA, STEP_ETA):
        cur_val, temp_v0, temp_v1 = log_likelihood_td(actions, rewards, eta, v0, v1)
        likelihood.append(cur_val)
        if cur_val > max_val:
            max_eta = eta
            max_val = cur_val
            final_v0 = temp_v0
            final_v1 = temp_v1
    return max_eta, final_v0, final_v1


def arg_max_likelihood_rein_all_etas(actions, rewards, w):
    max_eta = 0
    max_val = -np.infty
    final_w = w
    likelihood = []
    for eta in np.arange(MIN_ETA, MAX_ETA, STEP_ETA):
        cur_val, temp_w = log_likelihood_rein(actions, rewards, eta, w)
        likelihood.append(cur_val)
        if cur_val > max_val:
            max_eta = eta
            max_val = cur_val
            final_w = temp_w
    return max_eta, final_w, likelihood


def arg_max_likelihood_td_all_etas(actions, rewards, v0, v1):
    max_eta = 0
    max_val = -np.infty
    final_v0 = v0
    final_v1 = v1
    likelihood = []
    for eta in np.arange(MIN_ETA, MAX_ETA, STEP_ETA):
        cur_val, temp_v0, temp_v1 = log_likelihood_td(actions, rewards, eta, v0, v1)
        likelihood.append(cur_val)
        if cur_val > max_val:
            max_eta = eta
            max_val = cur_val
            final_v0 = temp_v0
            final_v1 = temp_v1
    return max_eta, final_v0, final_v1, likelihood

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
    max_eta, temp_v0, temp_v1 = arg_max_likelihood_td(second_actions, second_rewards, final_v0, final_v1)
    res.append(max_eta)
    return res


def analyze_participant_data_all_etas(name):
    first_act_file = open("first_actions_%s.txt" % name, 'r')
    first_actions = [int(line[0]) for line in first_act_file.readlines()]
    first_rewards_file = open("first_rewards_%s.txt" % name, 'r')
    first_rewards = [int(line[0]) for line in first_rewards_file.readlines()]
    second_act_file = open("second_actions_%s.txt" % name, 'r')
    second_actions = [int(line[0]) for line in second_act_file.readlines()]
    second_rewards_file = open("second_rewards_%s.txt" % name, 'r')
    second_rewards = [int(line[0]) for line in second_rewards_file.readlines()]

    res = []
    max_eta, final_w, likelihood_rein1 = arg_max_likelihood_rein_all_etas(first_actions, first_rewards, 0)
    res.append(max_eta)
    max_eta, temp_w, likelihood_rein2 = arg_max_likelihood_rein_all_etas(second_actions, second_rewards, final_w)
    res.append(max_eta)
    max_eta, final_v0, final_v1, likelihood_td1 = arg_max_likelihood_td_all_etas(first_actions, first_rewards, 0, 0)
    res.append(max_eta)
    max_eta, temp_v0, temp_v1, likelihood_td2 = arg_max_likelihood_td_all_etas(second_actions, second_rewards, final_v0, final_v1)
    res.append(max_eta)
    return res, likelihood_rein1, likelihood_rein2, likelihood_td1, likelihood_td2


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
    for i in range(NUMBER_OF_SIMULATIONS):
        simulation_actions, simulation_rewards = simulation_rein(eta1, eta2, P_1, P_0, t1, t2)
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
    for i in range(NUMBER_OF_SIMULATIONS):
        simulation_actions, simulation_rewards = simulation_td(eta1, eta2, P_1, P_0, t1, t2)
        first_actions = simulation_actions[:t1]
        second_actions = simulation_actions[t1:]
        first_rewards = simulation_rewards[:t1]
        second_rewards = simulation_rewards[t1:]

        max_eta, final_v0, final_v1 = arg_max_likelihood_td(first_actions, first_rewards, 0, 0)
        etas1.append(max_eta)
        max_eta, temp_v0, temp_v1 = arg_max_likelihood_td(second_actions, second_rewards, final_v0, final_v1)
        etas2.append(max_eta)
    return etas1, etas2


def histogram(data):
    # bins = np.linspace(-0.1, 1, 100)
    # plt.hist(data, bins, alpha=0.5)
    # plt.legend(loc='upper left')
    # plt.show()
    return sum(data)/len(data), statistics.variance(data)


def simulations_graph(x1, participant_y1, simulations_y1, simulations_e1, x2, participant_y2, simulations_y2, simulations_e2, model):
    plt.errorbar(x1, simulations_y1, yerr=simulations_e1, fmt='none', capsize=3, label='range of simulations estimated etas - 1st part')
    plt.scatter(x1, participant_y1, color='blue', label='participant estimated eta - 1st part')
    plt.errorbar(x2, simulations_y2, yerr=simulations_e2, fmt='none', ecolor='red', capsize=3, label='range of simulations estimated etas - 2nd part')
    plt.scatter(x2, participant_y2, color='red', label='participant estimated eta - 2nd part')
    plt.xlabel('participant')
    plt.ylabel('eta')
    plt.xticks(x1, x1)
    plt.legend(loc=10, bbox_to_anchor=(0.25, -0.6, 0.5, 0.5))
    plt.title("Estimated Etas of Participants and Simulations - " + model)
    plt.show()


def calculate_models_success():
    avg1rein = []
    avg2rein = []
    avg1td = []
    avg2td = []
    etas = []
    for eta in np.arange(MIN_ETA, MAX_ETA, STEP_ETA):
        etas.append(eta)
        sum1td = 0
        sum2td = 0
        sum1rein = 0
        sum2rein = 0
        for i in range(NUMBER_OF_SIMULATIONS):
            actions, rewards = simulation_td(eta, eta, P_1, P_0, DEFAULT_PART_LEN, DEFAULT_PART_LEN)
            sum1td += sum(rewards[:DEFAULT_PART_LEN])
            sum2td += sum(rewards[DEFAULT_PART_LEN:])
            actions, rewards = simulation_rein(eta, eta, P_1, P_0, DEFAULT_PART_LEN, DEFAULT_PART_LEN)
            sum1rein += sum(rewards[:DEFAULT_PART_LEN])
            sum2rein += sum(rewards[DEFAULT_PART_LEN:])
        avg1rein.append(sum1rein/NUMBER_OF_SIMULATIONS)
        avg2rein.append(sum2rein/NUMBER_OF_SIMULATIONS)
        avg1td.append(sum1td/NUMBER_OF_SIMULATIONS)
        avg2td.append(sum2td/NUMBER_OF_SIMULATIONS)

    return etas, avg1rein, avg2rein, avg1td, avg2td


def plot_models_success(etas, avg1rein, avg2rein, avg1td, avg2td):
    plt.plot(etas, avg2rein, color='blue', label="2nd part REINFORCE")
    plt.plot(etas, avg2td, color='red', label="2nd part TDL")
    plt.plot(etas, avg1rein, linestyle='dotted', color='blue', label="1st part REINFORCE")
    plt.plot(etas, avg1td, linestyle='dotted', color='red', label="1st part TDL")
    plt.xticks(etas, etas)
    plt.title("Average Sum of Rewards in Simulations")
    plt.xlabel('eta')
    plt.ylabel('number of rewards')
    plt.legend()
    plt.show()


def estimate_simulations(names, t1, t2, eta1rein, eta2rein,eta1td, eta2td):
    est_eta1_rein = []
    var_est_eta1_rein = []
    est_eta2_rein = []
    var_est_eta2_rein = []
    est_eta1_td = []
    var_est_eta1_td = []
    est_eta2_td = []
    var_est_eta2_td = []

    for i in range(len(names)):
        etas1, etas2 = big_simulation_rein(eta1rein[i], eta2rein[i], t1[i], t2[i])
        est_eta1_rein.append(sum(etas1)/len(etas1))
        var_est_eta1_rein.append(statistics.variance(etas1))
        est_eta2_rein.append(sum(etas2)/len(etas2))
        var_est_eta2_rein.append(statistics.variance(etas2))

        etas1, etas2 = big_simulation_td(eta1td[i], eta2td[i], t1[i], t2[i])
        est_eta1_td.append(sum(etas1)/len(etas1))
        var_est_eta1_td.append(statistics.variance(etas1))
        est_eta2_td.append(sum(etas2)/len(etas2))
        var_est_eta2_td.append(statistics.variance(etas2))

    print(est_eta1_rein)
    print(var_est_eta1_rein)
    print(est_eta2_rein)
    print(var_est_eta2_rein)
    print(est_eta1_td)
    print(var_est_eta1_td)
    print(est_eta2_td)
    print(var_est_eta2_td)
    return est_eta1_rein, var_est_eta1_rein, est_eta2_rein, var_est_eta2_rein, est_eta1_td, var_est_eta1_td, est_eta2_td, var_est_eta2_td


def restoration_process():
    eta1rein = []
    eta2rein = []
    eta1td = []
    eta2td = []
    for i, name in enumerate(NAMES):
        res = analyze_participant_data(name)
        eta1rein.append(res[0])
        eta2rein.append(res[1])
        eta1td.append(res[2])
        eta2td.append(res[3])
        print(name, res)
    est_eta1_rein, var_est_eta1_rein, est_eta2_rein, var_est_eta2_rein, est_eta1_td, var_est_eta1_td, est_eta2_td, var_est_eta2_td = estimate_simulations(
        NAMES, T1, T2, eta1rein, eta2rein, eta1td, eta2td)
    simulations_graph(PARTICIPANTS_NUMBERS, eta1rein, est_eta1_rein, var_est_eta1_rein,
                      SHIFTED_PARTICIPANTS_NUMBERS, eta2rein, est_eta2_rein, var_est_eta2_rein, "REINFORCE")

    simulations_graph(PARTICIPANTS_NUMBERS, eta1td, est_eta1_td, var_est_eta1_td,
                      SHIFTED_PARTICIPANTS_NUMBERS, eta2td, est_eta2_td, var_est_eta2_td, "TD")


def plot_all_etas():
    eta1rein = []
    eta2rein = []
    eta1td = []
    eta2td = []
    for i, name in enumerate(NAMES):
        res, likelihood_rein1, likelihood_rein2, likelihood_td1, likelihood_td2 = analyze_participant_data_all_etas(name)
        etas = np.arange(MIN_ETA, MAX_ETA, STEP_ETA)
        plt.plot(etas, likelihood_rein1, label="REINFORCE 1")
        plt.plot(etas, likelihood_rein2, label="REINFORCE 2")
        plt.plot(etas, likelihood_td1, label="TDL 1")
        plt.plot(etas, likelihood_td2, label="TDL 2")
        plt.title(f"Likelihood According to Eta - Participant {i+1}")
        # plt.ylim(-100, 10)
        plt.xlabel('eta')
        plt.ylabel('likelihood')
        plt.legend()
        plt.show()
        eta1rein.append(res[0])
        eta2rein.append(res[1])
        eta1td.append(res[2])
        eta2td.append(res[3])
        print(name, res)


def run_experiment():
    trials_a = 80
    trials_b = 80
    participant_id = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if len(sys.argv) > 1:
        participant_id += "_" + sys.argv[1]
    if len(sys.argv) > 2:
        trials_a = int(sys.argv[2])
    if len(sys.argv) > 3:
        trials_b = int(sys.argv[3])
    full_experiment(P_1, P_0, trials_a, trials_b, participant_id)


# restoration_process()
plot_all_etas()

