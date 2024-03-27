import numpy as np
import copy
import matplotlib.pyplot as plt
import utils
import gymnasium as gym
import types
import value_iter
import gauss_seidel
import policy_iteration
import prioritise_sweeping as ps

def optimal_values(env:gym.Env, iterations=5e5, epsilon=1e-3, gamma=0.9):
    optimal_values = { state: 0 for state in range(env.observation_space.n)}

    for _ in range(int(iterations)):

        max_threshold = -np.inf

        for state in range(env.observation_space.n):
            optimal_values_before = optimal_values[state]

            max_value = -np.inf

            for action in range(env.action_space.n):
                crt_reward = None
                crt_sum = 0

                for prob, next_state, reward, _ in env.P[state][action]:
                    if crt_reward is None:
                        crt_reward = reward
                    
                    crt_sum += prob * optimal_values[next_state]

                max_value = max(max_value, crt_reward + gamma * crt_sum)

            optimal_values[state] = max_value

            max_threshold = max(max_threshold, abs(optimal_values[state] - optimal_values_before))

        if max_threshold < epsilon:
            break

    return optimal_values

def policy_iteration_average_score(env, optimal_values, tries=5):
    iteration_values_tries = []
    crt_values_tries = []
    iteration_groundtruth_diff = []

    median_idx = tries // 2

    for _ in range(tries):
        crt_iteration_values, crt_values = policy_iteration.policy_iteration(env, optimal_values)
        iteration_values_tries.append(crt_iteration_values)
        crt_values_tries.append(crt_values)
        iteration_groundtruth_diff.append(compare_values(crt_values, optimal_values))


    indexes = np.argsort(iteration_groundtruth_diff)
    median_idx = indexes[median_idx]

    return iteration_values_tries[median_idx], crt_values_tries[median_idx]

def avg_score(env, optimal_values, tries=5):
    iteration_values_tries = []
    crt_values_tries = []
    iteration_groundtruth_diff = []

    median_idx = tries // 2

    for _ in range(tries):
        crt_iteration_values, crt_values = ps.prioritis_sweeping(env, optimal_values)
        iteration_values_tries.append(crt_iteration_values)
        crt_values_tries.append(crt_values)
        iteration_groundtruth_diff.append(compare_values(crt_values, optimal_values))


    indexes = np.argsort(iteration_groundtruth_diff)
    median_idx = indexes[median_idx]

    return iteration_values_tries[median_idx], crt_values_tries[median_idx]

def compare_values(v1, optimal_values):
    diff = 0
    for state in optimal_values:
        diff += abs(optimal_values[state] - v1[state])
    return diff

def pad_ys(ys):
    max_len = np.max([len(y) for y in ys])

    for i in range(len(ys)):
        crt_len = len(ys[i])
        if crt_len < max_len:
            ys[i] += [ys[i][-1]] * (max_len - crt_len)

    return ys

def values_plot(env:gym.Env):
    game_optimal_value = optimal_values(env)
    
    value_iter_values, _ = value_iter.value_iteration(env, game_optimal_value)
    gauss_seidel_values, _ = gauss_seidel.gauss_seidel_value_iter(env, game_optimal_value)
    policy_iter_values, _ = policy_iteration_average_score(env, game_optimal_value)
    priorites_values, _ = avg_score(env, game_optimal_value)
    
    ys = pad_ys([value_iter_values, policy_iter_values, gauss_seidel_values, priorites_values])
    
    
    return ys

def plot_algorithms(ys, names, env_name):
    plt.figure(figsize=(10, 10))

    max_y_len = np.max([len(y) for y in ys])
    
    x = np.arange(0, max_y_len)
    plot = plt.plot()
    for y, name in zip(ys, names):
        plt.plot(x, y, label=name)
    plt.title("Comparative optimal state value difference based on algorithm and iteration for {}".format(env_name))
    plt.xlabel("Iterations")
    plt.ylabel("||V - V*||")
    plt.legend()
    plt.show()
    return plot