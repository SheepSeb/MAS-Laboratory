import numpy as np
import copy
import matplotlib.pyplot as plt
import utils
import gymnasium as gym
import types

def gauss_seidel_value_iter(env:gym.Env, optimal_values, iterations:int = 5e5, epsilon:float = 1e-3, gamma:float = 0.9):
    crt_values = {state: 0 for state in range(env.observation_space.n)}
    
    iter_values = []
    
    iterations = int(iterations)
    
    for _ in range(iterations):
        max_threshold = -np.inf
        
        for state in range(env.observation_space.n):
            b_value = crt_values[state]
            max_value = -np.inf
            
            for act in range(env.action_space.n):
                b_reward = None
                b_sum = 0
                
                for prob, next_state, reward, _ in env.P[state][act]:
                    if b_reward is None:
                        b_reward = reward
                    b_sum += prob * crt_values[next_state]
                max_value = max(max_value, b_reward + gamma * b_sum)
            
            crt_values[state] = max_value
            iter_values.append(utils.compare_values(crt_values, optimal_values))
            
            max_threshold = max(max_threshold, abs(crt_values[state] - b_value))
        
        if max_threshold < epsilon:
            break
        
    return iter_values, crt_values