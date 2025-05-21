# Imports
import gym
import random
import numpy as np
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.utils.data as data

from Algorithms import forward_model

import d4rl

# Load environment
environment = 'Walker2d'
env_name = 'walker2d-medium-expert-v2'
env = gym.make(env_name)
dataset = d4rl.qlearning_dataset(env)

seed = 6
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Convert D4RL to replay buffer
print("Converting data...")
mean = np.mean(dataset["observations"], 0)
std = np.std(dataset["observations"], 0) + 1e-3
states = torch.Tensor((dataset["observations"] - mean) / std)
actions = torch.Tensor(dataset["actions"])
rewards = torch.Tensor(dataset["rewards"])
next_states = torch.Tensor((dataset["next_observations"] - mean) / std)
dones = torch.Tensor(dataset["terminals"])
replay_buffer = [states, actions, rewards, next_states, dones]
print("...data conversion complete")




# Convert data to DataLoader
print("Pre processing data...")
batch_size = 256
replay_buffer_env = []
for i in range(len(dataset["observations"])):
    if dataset["terminals"][i] == True:
        if (dataset["next_observations"][i] == dataset["observations"][i+1]).all():
            pass
        else:
            replay_buffer_env.append(((dataset["observations"][i] - mean) / std, dataset["actions"][i],
                                      dataset["rewards"][i],(dataset["next_observations"][i] - mean) / std,
                                      dataset["terminals"][i]))
    else:
        replay_buffer_env.append(((dataset["observations"][i] - mean) / std, dataset["actions"][i], dataset["rewards"][i],
                                  (dataset["next_observations"][i] - mean) / std, dataset["terminals"][i]))
random.shuffle(replay_buffer_env)
replay_buffer_train = data.DataLoader(replay_buffer_env[0: int(0.9 * len(replay_buffer_env))], batch_size=batch_size, shuffle=True)
replay_buffer_val = data.DataLoader(replay_buffer_env[int(0.9 * len(replay_buffer_env)):len(replay_buffer_env)], batch_size=batch_size, shuffle=True)
print("data processed!")
max_epochs = 1000
patience = 50 #20

# Hyperparameters and initialisation
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
device = "cuda:0"

det_dynamics = forward_model.DetDynamics(state_dim, action_dim, device=device)

# Training
val_loss = []
start = time.time()
for epoch in range(max_epochs):
    t, v = det_dynamics.train_dynamics(replay_buffer_train, replay_buffer_val)
    val_loss.append(v)
    if v == np.min(val_loss):
        torch.save(det_dynamics.dynamics.state_dict(), f"Models/{environment}/Forward_{env_name}_S{seed}.pt")
        best_model = epoch
        v_best = v
        print("Model improvement...saved", "Epoch", best_model, "Loss %.6f" % v_best)
    if np.sum(val_loss[-patience:] <= np.min(val_loss)) < 1:
        print("No improvement for", patience,  "updates.  Model best at epoch", best_model, "Model loss %.6f" % v_best)
        break
    print("Epoch", epoch, "Current loss %.6f" % v, "Best loss %.6f" % v_best)
end = time.time()
print("Total model training time", end-start)