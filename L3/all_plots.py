import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from copy import deepcopy
from queue import PriorityQueue
import sys
import random

NO_ITERATIONS = 5e5
GAMMA = 0.9
EPS = 1e-3
SEED = 0
tries_pi = 5

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

def plot_envs(ys_envs, names_envs, env_names):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
    plt.subplots_adjust(hspace=0.5, wspace=1)
    # fig = plt.figure()
    fig.suptitle("Results over all the environments", fontsize=18, y=0.95)

    for i, ax in enumerate(axes.ravel()):
        if i >= len(ys_envs):
            ax.plot()
            plt.show()
            return

        nrows = i // 2
        ncols = i % 2
        axes[nrows, ncols].plot()

        ys_crt = ys_envs[i]
        names_crt = names_envs[i]
        env_name = env_names[i]

        
        x = np.arange(0, len(ys_crt[0]))

    #     ax.plot()
        for y, name in zip(ys_crt, names_crt):
            axes[nrows, ncols].plot(x, y, label=name)
            plot_title = "Comparative optimal state value difference based on algorithm and iteration for {}".format(env_name)
            axes[nrows, ncols].set_title(env_name)
            axes[nrows, ncols].set_xlabel("Iterations")
            axes[nrows, ncols].set_ylabel("||V - V*||")
            axes[nrows, ncols].legend()
        
    #     # plt.show()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("||V - V*||")

    plt.show()

    return
    
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


def optimal_values(env, iterations=5e5, epsilon=1e-3, gamma=0.9):
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

def value_iteration(env, optimal_values, iterations=5e5, epsilon=1e-3, gamma=0.9):
    crt_values = { state: 0 for state in range(env.observation_space.n)}

    iterations_values = []
    
    for _ in range(int(iterations)):

        max_threshold = -np.inf

        prev_values = deepcopy(crt_values)

        for state in range(env.observation_space.n):
            before_value = crt_values[state]

            max_value = -np.inf

            for action in range(env.action_space.n):
                crt_reward = None
                crt_sum = 0

                for prob, next_state, reward, _ in env.P[state][action]:
                    if crt_reward is None:
                        crt_reward = reward
                    
                    crt_sum += prob * prev_values[next_state]

                max_value = max(max_value, crt_reward + gamma * crt_sum)

            crt_values[state] = max_value
            iterations_values.append(compare_values(crt_values, optimal_values))

            max_threshold = max(max_threshold, abs(crt_values[state] - before_value))

        if max_threshold < epsilon:
            break

    return iterations_values, crt_values

def policy_iteration(env, optimal_values, iterations=5e5, epsilon=1e-3, gamma=0.9):
    crt_values = { state: 0 for state in range(env.observation_space.n)}
    crt_policy = { state: random.randint(0,env.action_space.n-1) for state in range(env.observation_space.n)}

    iterations_values = []

    for _ in range(int(iterations)):

        max_threshold = -np.inf

        for state in range(env.observation_space.n):
            previous_value = crt_values[state]
            possible_next_states = env.P[state][crt_policy[state]]

            crt_reward = None
            crt_sum = 0
            
            for prob, next_state, reward, _ in possible_next_states:
                if crt_reward is None:
                    crt_reward = reward
                
                crt_sum += prob * crt_values[next_state]
            
            crt_values[state] = crt_reward + gamma * crt_sum
            iterations_values.append(compare_values(crt_values, optimal_values))
            crt_threshold = abs(crt_values[state] - previous_value)
            max_threshold = max(max_threshold, crt_threshold)

        if max_threshold < epsilon:
            break

        for state in range(env.observation_space.n):
            best_action = None
            max_value = -np.inf

            for action in range(env.action_space.n):
                crt_reward = None
                crt_sum = 0

                for prob, next_state, reward, _ in env.P[state][action]:
                    if crt_reward is None:
                        crt_reward = reward
                    
                    crt_sum += prob * crt_values[next_state]

                crt_value = crt_reward + gamma * crt_sum
                if crt_value > max_value:
                    max_value = crt_value
                    best_action = action
                
            crt_policy[state] = best_action
            
    return iterations_values, crt_values

def policy_iteration_average_score(env, optimal_values, tries=5, iterations=5e5, epsilon=1e-3, gamma=0.9):
    iteration_values_tries = []
    crt_values_tries = []
    iteration_groundtruth_diff = []

    median_idx = tries // 2

    for _ in range(tries):
        crt_iteration_values, crt_values = policy_iteration(env, optimal_values, iterations=iterations, epsilon=epsilon, gamma=gamma)
        iteration_values_tries.append(crt_iteration_values)
        crt_values_tries.append(crt_values)
        iteration_groundtruth_diff.append(compare_values(crt_values, optimal_values))


    indexes = np.argsort(iteration_groundtruth_diff)
    median_idx = indexes[median_idx]

    return iteration_values_tries[median_idx], crt_values_tries[median_idx]

def gauss_seidel_value_iteration(env, optimal_values, iterations=5e5, epsilon=1e-3, gamma=0.9):
    crt_values = { state: 0 for state in range(env.observation_space.n)}

    iterations_values = []
    
    for _ in range(int(iterations)):

        max_threshold = -np.inf

        for state in range(env.observation_space.n):
            before_value = crt_values[state]

            max_value = -np.inf

            for action in range(env.action_space.n):
                crt_reward = None
                crt_sum = 0

                for prob, next_state, reward, _ in env.P[state][action]:
                    if crt_reward is None:
                        crt_reward = reward
                    
                    crt_sum += prob * crt_values[next_state]

                max_value = max(max_value, crt_reward + gamma * crt_sum)

            crt_values[state] = max_value
            iterations_values.append(compare_values(crt_values, optimal_values))

            max_threshold = max(max_threshold, abs(crt_values[state] - before_value))

        if max_threshold < epsilon:
            break

    return iterations_values, crt_values

def preprocess_states(env):
    dict_states = {state: (set([]), 0) for state in range(env.observation_space.n)}

    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            for _, next_state, _, _ in env.P[state][action]:
                dict_states[next_state][0].add(state)
        
        dict_states[state] = (dict_states[state][0], len(dict_states[state][0]))
    
    topological_order = []
    stack = [0]

    while len(stack) > 0:
        crt_state = stack.pop()

        if crt_state not in topological_order:
            topological_order.append(crt_state)

        for action in range(env.action_space.n):
            for _, next_state, _, _ in env.P[crt_state][action]:
                if next_state not in topological_order and next_state not in stack:
                    stack.append(next_state)


    for state in topological_order:
        predecessors = dict_states[state][0]
        max_priority = dict_states[state][1]

        for predecessor in predecessors:
            dict_states[state] = (dict_states[state][0], max(1 + dict_states[predecessor][1], max_priority))

    return dict_states

def max_error(queue):
    max_diff = -np.inf

    for error, _ in queue:
        max_diff = max(max_diff, abs(error))
    
    return max_diff

def argmax_heap(dictionary):
    max_value = -np.inf
    max_key = None

    for key, value in dictionary.items():
        if value > max_value:
            max_value = value
            max_key = key
    
    return max_key, max_value

## TODO De actualizat pe tine insuti, si pastrat in curent_values ulterior
def prioritized_sweeping_value_iteration(env, optimal_values, preprocess_dict, iterations=5e5, epsilon=1e-3, gamma=0.9):
    initialise_value = 10
    iterations_values, crt_values = value_iteration(env, optimal_values, iterations=initialise_value, epsilon=epsilon, gamma=gamma)

    heap = {state: crt_values[state] for state in range(env.observation_space.n)}

    for _ in range(int(iterations) - initialise_value):
        previous_values = deepcopy(crt_values)

        crt_state, max_error = argmax_heap(heap)

        previous_val = previous_values[crt_state]

        if max_error < epsilon:
            break

        max_value = -np.inf

        for action in range(env.action_space.n):
            crt_reward = None
            crt_sum = 0

            for prob, next_state, reward, _ in env.P[crt_state][action]:
                if crt_reward is None:
                    crt_reward = reward
                    
                crt_sum += prob * crt_values[next_state]

            max_value = max(max_value, crt_reward + gamma * crt_sum)

        heap[crt_state] = abs(max_value - previous_val)
        crt_values[crt_state] = max_value
        
        iterations_values.append(compare_values(crt_values, optimal_values))

        predecessors = preprocess_dict[crt_state][0]

        for predecessor in predecessors:

            previous_val_pred = crt_values[predecessor]
            
            max_value_pred = -np.inf

            for action in range(env.action_space.n):
                pred_reward = None
                pred_sum = 0

                for prob, next_state, reward, _ in env.P[crt_state][action]:
                    if pred_reward is None:
                        pred_reward = reward

                    if next_state == crt_state:    
                        pred_sum += prob * max_value
                    else:
                        pred_sum += prob * crt_values[next_state]

                max_value_pred = max(max_value_pred, pred_reward + gamma * pred_sum)

            heap[predecessor] = abs(max_value_pred - previous_val_pred)
    return iterations_values, crt_values

def obtain_values_for_plots(env, iterations=5e5, epsilon=1e-3, gamma=0.9):
    game_optimal_values = optimal_values(env, iterations=NO_ITERATIONS, epsilon=EPS, gamma=GAMMA)

    preprocess_dict = preprocess_states(env)

    value_iteration_diffs, _ = value_iteration(env, game_optimal_values, iterations=NO_ITERATIONS, epsilon=EPS, gamma=GAMMA)
    # policy_iteration_diffs, _ = policy_iteration(env, game_optimal_values, iterations=NO_ITERATIONS, epsilon=EPS, gamma=GAMMA)
    policy_iteration_diffs, _ = policy_iteration_average_score(env, game_optimal_values, tries=tries_pi, iterations=NO_ITERATIONS, epsilon=EPS, gamma=GAMMA)
    gauss_seidel_diffs, _ = gauss_seidel_value_iteration(env, game_optimal_values, iterations=NO_ITERATIONS, epsilon=EPS, gamma=GAMMA)
    prioritized_sweeping_diffs, _ = prioritized_sweeping_value_iteration(env, game_optimal_values, preprocess_dict, iterations=1000, epsilon=EPS, gamma=GAMMA)

    ys = pad_ys([value_iteration_diffs, policy_iteration_diffs, gauss_seidel_diffs, prioritized_sweeping_diffs])

    return ys

if __name__ == "__main__":
    env_names:list[str] = ["Taxi-v3","FrozenLake-v1", "FrozenLake8x8-v1"]
    algorithms:list[str] = ["value_iteration", "policy_iteration", "gauss_seidel", "prioritise_sweeping"]

    all_ys = []

    for env_name in env_names:
        env = gym.make(env_name)    
        observation, info = env.reset(seed=SEED)
    
        ys = obtain_values_for_plots(env, iterations=NO_ITERATIONS, epsilon=EPS, gamma=GAMMA)
        # plot_algorithms(ys, names, env_name=env_name)
        all_ys.append(ys)
    
    plot_envs(all_ys, 3*[algorithms], env_names)

    # https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d
    # https://gymnasium.farama.org/environments/toy_text/taxi/#taxi