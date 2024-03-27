import numpy as np
import copy
import matplotlib.pyplot as plt
import utils
import gymnasium as gym
import types
import queue

def prioritis_sweeping(env:gym.Env, optimal_values, iterations:int = 5e5, epsilon:float = 1e-3, gamma:float = 0.9):
    crt_values = {state: 0 for state in range(env.observation_space.n)}
    
    p_q = queue.PriorityQueue()
    # Get all the states of the environment
    all_states = range(env.observation_space.n)
    # Init V(s) = 0 for all states
    for state in all_states:
        p_q.put((-np.inf, state))
        
    iter_values = []
    
    iterations = int(iterations)
    
    for _ in range(iterations):
        # Get the state with the highest priority
        _, state = p_q.get()
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
        # print(crt_values)
        iter_values.append(utils.compare_values(crt_values, optimal_values))
        
        max_threshold = abs(crt_values[state] - b_value)
        # print(max_threshold)
        p_q.put((-max_threshold, state))
        
        for act in range(env.action_space.n):
            for prob, next_state, _, _ in env.P[state][act]:
                if next_state != state:
                    p_q.put((-max_threshold, next_state))
                    
        if max_threshold < epsilon:
            break
        
        # print(p_q.queue)
    
    return iter_values, crt_values