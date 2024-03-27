import numpy as np
import copy
import matplotlib.pyplot as plt
import utils
import gymnasium as gym
import types
import random

def policy_iteration(env:gym.Env, optimal_values, iterations:float = 5e5, epsilon:float = 1e-3, gamma:float = 0.9):
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
            iterations_values.append(utils.compare_values(crt_values, optimal_values))
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