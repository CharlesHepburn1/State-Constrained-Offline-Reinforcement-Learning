# Imports
import gym
import random
import numpy as np
import copy
import pickle
import time
import torch
import d4rl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from Algorithms import StaCQ_multistep
from Algorithms import forward_model
from Algorithms import inverse_model

# Load environment
environment = 'Halfcheetah'
env_name = 'halfcheetah-medium-expert-v2'
env = gym.make(env_name)
dataset = d4rl.qlearning_dataset(env)
device = 'cuda:0'
policy_extract = 'StateBCReg'
num_critics = 4
policy_noise_act = 0.1  #0.01, 0.1, 0.2
Actor_ens_min = True  #False
independent_target = False
hyp = 0.5 #change this dependendent on environment

seeds = [1,2,3,4,5]
discount_factor = 0.99
print("Converting data...")
st_mean = np.mean(np.array(dataset["observations"]), 0)
st_std = np.std(np.array(dataset["observations"]), 0) +1e-3

states = np.array((dataset["observations"]-st_mean)/st_std)
actions = np.array(dataset["actions"])
rewards = np.array(dataset["rewards"])
next_states = np.array((dataset["next_observations"]-st_mean)/st_std)

dones = np.array(dataset["terminals"])

done_traj = np.zeros((len(states),))
for i in range(len(states)):
    statement1 = (dataset["terminals"][i] == 1.0)
    if i == len(states)-1:
        statement2 = True
    else:
        statement2 = (dataset["observations"][i + 1] != dataset["next_observations"][i]).all()
    if statement1 or statement2:
        done_traj[i]= 1.0
    else:
        done_traj[i] = 0.0

done_traj = torch.Tensor(done_traj)

print("...data conversion complete")


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]

f_sa0 = forward_model.Dynamics(state_dim, action_dim).to(device)
f_sa0.load_state_dict(torch.load(f"Models/{environment}/Forward_{env_name}_S0.pt",  map_location=device))
f_sa1 = forward_model.Dynamics(state_dim, action_dim).to(device)
f_sa1.load_state_dict(torch.load(f"Models/{environment}/Forward_{env_name}_S1.pt",  map_location=device))
f_sa2 = forward_model.Dynamics(state_dim, action_dim).to(device)
f_sa2.load_state_dict(torch.load(f"Models/{environment}/Forward_{env_name}_S2.pt",  map_location=device))
f_sa3 = forward_model.Dynamics(state_dim, action_dim).to(device)
f_sa3.load_state_dict(torch.load(f"Models/{environment}/Forward_{env_name}_S3.pt",  map_location=device))
f_sa4 = forward_model.Dynamics(state_dim, action_dim).to(device)
f_sa4.load_state_dict(torch.load(f"Models/{environment}/Forward_{env_name}_S4.pt",  map_location=device))
f_sa5 = forward_model.Dynamics(state_dim, action_dim).to(device)
f_sa5.load_state_dict(torch.load(f"Models/{environment}/Forward_{env_name}_S5.pt",  map_location=device))
f_sa6 = forward_model.Dynamics(state_dim, action_dim).to(device)
f_sa6.load_state_dict(torch.load(f"Models/{environment}/Forward_{env_name}_S6.pt",  map_location=device))

f_sa = [f_sa0,f_sa1,f_sa2, f_sa3, f_sa4, f_sa5, f_sa6]

inv_ss0 = inverse_model.InverseModel(state_dim=state_dim, action_dim=action_dim).to(device)
inv_ss0.load_state_dict(torch.load(f"Models/{environment}/Inverse_{env_name}_S0.pt",  map_location=device))
inv_ss1 = inverse_model.InverseModel(state_dim=state_dim, action_dim=action_dim).to(device)
inv_ss1.load_state_dict(torch.load(f"Models/{environment}/Inverse_{env_name}_S1.pt",  map_location=device))
inv_ss2 = inverse_model.InverseModel(state_dim=state_dim, action_dim=action_dim).to(device)
inv_ss2.load_state_dict(torch.load(f"Models/{environment}/Inverse_{env_name}_S2.pt",  map_location=device))

inv_ss = [inv_ss0, inv_ss1, inv_ss2]

eps_X = 0.1
open_file = open(f"Reachability_indexes/StaCQ_BC/{environment}/Reachable_StaCQ_BC_{eps_X}_{env_name}_NS.pkl", "rb")
list_idx = pickle.load(open_file)
open_file.close()
list_idx = np.array(list_idx, dtype='object')

states = torch.Tensor(np.array(states))
actions = torch.Tensor(np.array(actions))
rewards = torch.Tensor(np.array(rewards))

if environment == 'Halfcheetah':
    pass
else:
    idx_terminals = np.where(dones ==1)[0]
    if idx_terminals[-1] !=len(states) -1:
        idx_terminals = np.append(idx_terminals, len(states) -1)

    for i in range(len(idx_terminals)):
        st = torch.Tensor(states[idx_terminals[i]].reshape(1,state_dim)).to(device)
        act = torch.Tensor(actions[idx_terminals[i]].reshape(1,action_dim)).to(device)
        delta_s0 = f_sa0(st, act).detach().cpu().numpy()
        delta_s1 = f_sa1(st, act).detach().cpu().numpy()
        delta_s2 = f_sa2(st, act).detach().cpu().numpy()
        delta_s3 = f_sa3(st, act).detach().cpu().numpy()
        delta_s4 = f_sa4(st, act).detach().cpu().numpy()
        delta_s5 = f_sa5(st, act).detach().cpu().numpy()
        delta_s6 = f_sa6(st, act).detach().cpu().numpy()
        delta_s = torch.Tensor(np.mean((delta_s0, delta_s1, delta_s2, delta_s3, delta_s4, delta_s5, delta_s6),
                                       axis=0)).to(device)
        next_states[idx_terminals[i]] = torch.Tensor(st.detach().cpu().numpy() + delta_s.detach().cpu().numpy())

next_states = torch.Tensor(np.array(next_states))

dones = torch.Tensor(np.array(dones))
done_traj = torch.Tensor(np.array(done_traj))
replay_buffer = [states, actions, rewards, next_states, dones, done_traj]

print("Dataset:", env_name, "Num_critics:", num_critics, "PolicyNoiseInActor:", policy_noise_act,
      "TakeMinofCriticsinActor", Actor_ens_min, "LambdaParam:", hyp )

for se in range(len(seeds)):
    start = time.time()
    seed = seeds[se]
    agent = StaCQ_multistep.Agent(state_dim, action_dim, max_action,policy_noise_act = policy_noise_act, gamma=discount_factor,lmbda = hyp,ind_targ= independent_target,
                                Actor_funct = policy_extract,Actor_ens_min = Actor_ens_min,Ens_size = num_critics, device=device)
    st_test = torch.Tensor(states[:100]).to(device)
    act_test = torch.Tensor(actions[:100]).to(device)
    nst_test = torch.Tensor(next_states[:100]).to(device)

    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    grad_steps = 0
    evals = 10
    epochs = 200
    iterations = 5000

    scores_train = []

    for epoch in range(epochs):
        # Training #
        agent.train2(replay_buffer=replay_buffer, list_idx = list_idx, forward_model=f_sa, inverse_model=inv_ss,
                    iterations=iterations)
        grad_steps += iterations
        for eval in range(evals):
            done = False
            state = env.reset()
            score = 0
            while not done:
                with torch.no_grad():
                    state = (state - st_mean) / st_std
                    action = agent.choose_action(state)
                    state, reward, done, info = env.step(action)
                    score += reward
            score_norm = env.get_normalized_score(score) * 100
            scores_train.append(score_norm)
    
    end = time.time()
    # Final policy eval
    evals = 10
    score_norm_history = []
    env = gym.make(env_name)
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
                state = (state-st_mean)/st_std
                action = agent.choose_action(state)
                # action = agent.select_action_a(state)
                state, reward, done, info = env.step(action)
                score += reward
        score_norm = env.get_normalized_score(score)*100
        score_norm_history.append(score_norm)
        # print("Eval", eval, "Score %.2f" % score_norm)


    print("Seed %.1f" %seed ,"Mean Score Norm %.1f" % np.mean(score_norm_history), "Std Score Norm %.1f" % np.std(score_norm_history))

    np.save(f'Learning_Curves/{env_name}_LC_{seed}.npy', scores_train)
    print("Time taken =%.4f " %(end-start), "seconds")

    torch.save(agent.actor.state_dict(),f"Final_Policy/Actor_{env_name}-S{seed}.pt")
    # torch.save(agent.critic.state_dict(),f"Models/{environment}/StaCQ_BC/Critic_{env_name}-S{seed}")
