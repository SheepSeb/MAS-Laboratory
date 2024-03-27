import numpy as np
import copy
import matplotlib.pyplot as plt
import utils
import gymnasium as gym
import types

if __name__ == "__main__":
    env_names:list[str] = ["Taxi-v3","FrozenLake-v1", "FrozenLake8x8-v1"]
    algorithms:list[str] = ["value_iteration", "policy_iteration", "gauss_seidel"]
    
    values_ys = []
    
    for env_name in env_names:
        env = gym.make(env_name)
        ys = utils.values_plot(env)
        utils.plot_algorithms(ys, algorithms, env_name)