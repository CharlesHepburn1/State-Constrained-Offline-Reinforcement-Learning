# Imports
import gym
import random
import numpy as np
import copy
import pickle

import torch
import d4rl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import StaCQ_Multistep_algo as StaCQ
import forward_model
import inverse_model
import torch.utils.data as data
import time


# Load environment
environment = 'Antmaze' #'Halfcheetah'
env_name = 'antmaze-large-diverse-v2' #"halfcheetah-expert-v2"
env = gym.make(env_name)
# dataset = d4rl.qlearning_dataset(env)
device = 'cuda:0'
reward_func =  1
discount_factor = 0.99
num_critics = 4
hyp = 19 # 10.0 #1.0  #0.1 #1.0 #0.3  #[0.1, 0.3, 1.0, 3.0, 10.0, 30.0] #  1/0.1 = 10 is the hyp for STR


seeds =[1,2,3,4,5]

############### Normalized #####
normalise = False

env = gym.make(env_name)
dataset = d4rl.qlearning_dataset(env)
# env.seed(seed)
# env.action_space.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]

if env_name == 'antmaze-umaze-diverse-v2':
    print("1")
    states = dataset["observations"]
    actions = dataset["actions"]
    ## Get approximate original goal by looking at rewards
    rewards = dataset["rewards"]
    goal_reached = states[:, :2][rewards == 1]
    orig_goal = np.mean(goal_reached, 0).reshape(1, -1)
    orig_goals = np.repeat(orig_goal, len(states), 0)

    threshold = 0.5
    xy = states[:, :2]
    distances = np.linalg.norm(xy - orig_goals, axis=-1)
    at_goal = distances < threshold
    dones = torch.Tensor(at_goal)
    rewards = torch.Tensor(at_goal) - 1

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
else:
    states = torch.Tensor(np.array(dataset["observations"]))
    actions = torch.Tensor(np.array(dataset["actions"]))
    rewards = torch.Tensor(np.array(dataset["rewards"])) - 1
    dones = torch.Tensor(np.array(dataset["terminals"]))

next_states = torch.Tensor(np.array(dataset["next_observations"]))
replay_buffer = [states, actions, rewards, next_states, dones]


f_sa0 = forward_model.Dynamics2(state_dim, action_dim).to(device)
f_sa0.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Forward_{env_name}_S0.pt",  map_location=device))
f_sa1 = forward_model.Dynamics2(state_dim, action_dim).to(device)
f_sa1.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Forward_{env_name}_S1.pt",  map_location=device))
f_sa2 = forward_model.Dynamics2(state_dim, action_dim).to(device)
f_sa2.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Forward_{env_name}_S2.pt",  map_location=device))
f_sa3 = forward_model.Dynamics2(state_dim, action_dim).to(device)
f_sa3.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Forward_{env_name}_S3.pt",  map_location=device))
f_sa4 = forward_model.Dynamics2(state_dim, action_dim).to(device)
f_sa4.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Forward_{env_name}_S4.pt",  map_location=device))
f_sa5 = forward_model.Dynamics2(state_dim, action_dim).to(device)
f_sa5.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Forward_{env_name}_S5.pt",  map_location=device))
f_sa6 = forward_model.Dynamics2(state_dim, action_dim).to(device)
f_sa6.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Forward_{env_name}_S6.pt",  map_location=device))

f_sa = [f_sa0, f_sa1, f_sa2, f_sa3, f_sa4, f_sa5, f_sa6]

inv_ss0 = inverse_model.InverseModel2(state_dim=state_dim, action_dim=action_dim, max_action = max_action).to(device)
inv_ss0.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Inverse_{env_name}_S0.pt",  map_location=device))
inv_ss1 = inverse_model.InverseModel2(state_dim=state_dim, action_dim=action_dim, max_action = max_action).to(device)
inv_ss1.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Inverse_{env_name}_S1.pt",  map_location=device))
inv_ss2 = inverse_model.InverseModel2(state_dim=state_dim, action_dim=action_dim, max_action = max_action).to(device)
inv_ss2.load_state_dict(torch.load(f"Models/Test2_UnNormalised_Inverse_{env_name}_S2.pt",  map_location=device))

inv_ss = [inv_ss0, inv_ss1, inv_ss2]

eps_X = 0.1
open_file = open(f"Reachability_indexes/Reachable_StaCQ_BC_{eps_X}_{env_name}_NS.pkl", "rb")
list_idx = pickle.load(open_file)
open_file.close()
list_idx = np.array(list_idx, dtype='object')


#agent = StaCQ.Agent(state_dim, action_dim, max_action, gamma=discount_factor,lmbda = hyp,device=device)
#st_test = torch.Tensor(states[:100]).to(device)
#act_test = torch.Tensor(actions[:100]).to(device)
#nst_test = torch.Tensor(next_states[:100]).to(device)

grad_steps = 0
evals = 100
epochs = 200
iterations = 5000 #5000


for i in range(len(seeds)):
    seed = seeds[i]
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = StaCQ.Agent(state_dim, action_dim, max_action,gamma = discount_factor,  batch_size=256,Ens_size = num_critics,
            policy_noise = 0.2,policy_noise_act = 0.1,noise_clip_act=0.5, lmbda = hyp, policy_freq = 1,
            ind_targ = True, Actor_funct = 'StateBCReg',device=device)
    scores_train =[]
    for epoch in range(epochs):
        agent.train2(replay_buffer,list_idx = list_idx, forward_model=f_sa,inverse_model=inv_ss,  iterations=iterations)
        grad_steps += iterations
        score_tmp = []
        for eval in range(evals):
            done = False
            state = env.reset()
            score = 0
            while not done:
                with torch.no_grad():
                    action = agent.choose_action(state)
                    # action = agent.select_action_a(state)
                    state, reward, done, info = env.step(action)
                    score += reward
            score_norm = env.get_normalized_score(score) * 100
            score_tmp.append(score_norm)
        scores_train.append(np.mean(score_tmp))
                #agent.train_simple(replay_buffer, forward_model= f_sa,iterations = iterations)
        # Final policy eval
    evals = 100
    score_norm_history = []
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    for eval in range(evals):
        done = False
        state = env.reset()
        score = 0
        while not done:
            with torch.no_grad():
                # if normalise:
                #     state = (state - st_mean) / st_std
                action = agent.choose_action(state)
                # action = agent.select_action_a(state)
            state, reward, done, info = env.step(action)
            score += reward
        score_norm = env.get_normalized_score(score)*100
        score_norm_history.append(score_norm)

    print("Seed %.1f" % seed,"Mean Score Norm %.1f" % np.mean(score_norm_history), "Std Score Norm %.1f" % np.std(score_norm_history))

    np.save(f'Learning_Curves/{env_name}_LC_{seed}.npy', scores_train)
    # print("Time taken =%.4f " % (end - start), "seconds")

    torch.save(agent.actor.state_dict(), f"Final_Policy/Actor_{env_name}-S{seed}.pt")

# torch.save(agent.stateactor.state_dict(),f"Models/{environment}/StaCQ_BC/Actor_{env_name}-S{seed}")
# torch.save(agent.critic.state_dict(keep_vars=True), f"Models/StaCQ/critic_StaCQ_{env_name}_S{seed}.pt")
# torch.save(agent.Rew.state_dict(keep_vars=True), f"Models/StaCQ/reward_StaCQ_{env_name}_S{seed}.pt")
# torch.save(agent.actor.state_dict(),f"Models/StaCQ/Actor_ExpWeight_lmbda_{hyp}_{env_name}-S{seed}.pt")
