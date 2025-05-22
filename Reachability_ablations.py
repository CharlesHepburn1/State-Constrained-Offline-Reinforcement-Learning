## Imports
import gym
import random
import numpy as np
import copy
import time
import itertools
import torch
from Algorithms import forward_model
from Algorithms import inverse_model
import pickle
import d4rl
from rtree import index
import sys


# Load environment

environment = 'Walker2d' 
env_name =  'walker2d-medium-v2' 
norm = 'linf' #'l2'    #'l1' 
load_minmax = False
# build_rtree = False
delta_X = 0.1 


env = gym.make(env_name)
# Hyperparameters
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_action = env.action_space.low[0]
max_action = env.action_space.high[0]

dataset = d4rl.qlearning_dataset(env)

seed = 42
offset = 100
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Convert D4RL to replay buffer
print("Converting data...")
st_mean = np.mean(np.array(dataset["observations"]), 0)
st_std = np.std(np.array(dataset["observations"]), 0) +1e-3

states = torch.Tensor((dataset["observations"]-st_mean)/st_std)
actions = torch.Tensor(dataset["actions"])
rewards = torch.Tensor(dataset["rewards"])
next_states = torch.Tensor((dataset["next_observations"]-st_mean)/st_std)
dones = torch.Tensor(dataset["terminals"])
# replay_buffer = [states, actions, rewards, next_states, dones]
print("...data conversion complete")

done_traj = np.zeros((len(states),1))
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

device = "cuda:0"
action = []
n_actions = 1000
for j in range(n_actions):
    action.append(env.action_space.sample())
action = torch.Tensor(action).to(device)

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

inv_ss0 = inverse_model.InverseModel(state_dim=state_dim, action_dim=action_dim).to(device)
inv_ss0.load_state_dict(torch.load(f"Models/{environment}/Inverse_{env_name}_S0.pt",  map_location=device))

inv_ss1 = inverse_model.InverseModel(state_dim=state_dim, action_dim=action_dim).to(device)
inv_ss1.load_state_dict(torch.load(f"Models/{environment}/Inverse_{env_name}_S1.pt",  map_location=device))

inv_ss2 = inverse_model.InverseModel(state_dim=state_dim, action_dim=action_dim).to(device)
inv_ss2.load_state_dict(torch.load(f"Models/{environment}/Inverse_{env_name}_S2.pt",  map_location=device))
inv_ss = [inv_ss0, inv_ss1, inv_ss2]


idx_terminals = np.where(dones ==1)[0]

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
replay_buffer = [states, actions, rewards, next_states, dones]


if not load_minmax:
    start = time.time()
    N = 1000
    actions_rep = action.repeat(N, 1)
    maxes = [[]*(state_dim)]*len(states)
    mins = [[]*(state_dim)]*len(states)
    for j in range(int((len(states)-1)/N) ):
        st = torch.Tensor(states[N*j:N*(j+1)]).to(device)
        states_rep = torch.repeat_interleave(st, n_actions, dim=0)
        delta_s0 = f_sa0(states_rep, actions_rep).detach().cpu().numpy()
        delta_s1 = f_sa1(states_rep, actions_rep).detach().cpu().numpy()
        delta_s2 = f_sa2(states_rep, actions_rep).detach().cpu().numpy()
        delta_s3 = f_sa3(states_rep, actions_rep).detach().cpu().numpy()
        delta_s4 = f_sa4(states_rep, actions_rep).detach().cpu().numpy()
        delta_s5 = f_sa5(states_rep, actions_rep).detach().cpu().numpy()
        delta_s6 = f_sa6(states_rep, actions_rep).detach().cpu().numpy()

        delta_s = torch.Tensor(np.mean((delta_s0, delta_s1, delta_s2, delta_s3, delta_s4, delta_s5, delta_s6),
                                       axis=0)).to(device)
        nst = (states_rep + delta_s).reshape(N, n_actions, state_dim)

        nst_max, _ = torch.max(nst, dim=1)
        nst_min, _ = torch.min(nst, dim=1)

        nst_max = nst_max.detach().cpu().numpy()
        nst_min = nst_min.detach().cpu().numpy()
        actual_nst = next_states[N*j:N*(j+1)].numpy()
        nst_max = np.maximum(nst_max, actual_nst+1e-6)
        nst_min = np.minimum(nst_min, actual_nst-1e-6)

        maxes[N*j:N*(j+1)] = nst_max
        mins[N*j:N*(j+1)] = nst_min

    uu = len(states) -int((len(states)-1)/N)*1000
    if uu >0:
        st = torch.Tensor(states[-uu:]).to(device)
        states_rep = torch.repeat_interleave(st, n_actions, dim=0)
        actions_rep2 = action.repeat(uu, 1)
        delta_s0 = f_sa0(states_rep, actions_rep2).detach().cpu().numpy()
        delta_s1 = f_sa1(states_rep, actions_rep2).detach().cpu().numpy()
        delta_s2 = f_sa2(states_rep, actions_rep2).detach().cpu().numpy()
        delta_s3 = f_sa3(states_rep, actions_rep2).detach().cpu().numpy()
        delta_s4 = f_sa4(states_rep, actions_rep2).detach().cpu().numpy()
        delta_s5 = f_sa5(states_rep, actions_rep2).detach().cpu().numpy()
        delta_s6 = f_sa6(states_rep, actions_rep2).detach().cpu().numpy()

        delta_s = torch.Tensor(np.mean((delta_s0, delta_s1, delta_s2, delta_s3, delta_s4, delta_s5,delta_s6),
                                       axis=0)).to(device)
        nst = (states_rep + delta_s).reshape(uu, n_actions, state_dim)

        nst_max, _ = torch.max(nst, dim=1)
        nst_min, _ = torch.min(nst, dim=1)

        nst_max = nst_max.detach().cpu().numpy()
        nst_min = nst_min.detach().cpu().numpy()
        actual_nst = next_states[-uu:].numpy()
        nst_max = np.maximum(nst_max, actual_nst+1e-6)
        nst_min = np.minimum(nst_min, actual_nst-1e-6)

        maxes[-uu:] = nst_max
        mins[-uu:] = nst_min
    end = time.time()
    print("Time taken = ", end - start, "seconds")

    np.save(f"Reachability_indexes/Ablations/Mins_{env_name}",mins)
    np.save(f"Reachability_indexes/Ablations/Maxs_{env_name}",maxes)
else:
    mins = np.load(f"Reachability_indexes/Ablations/Mins_{env_name}.npy")
    maxes = np.load(f"Reachability_indexes/Ablations/Maxs_{env_name}.npy")

SorNS = 'NS' #'S'
if SorNS == 'S':
    states_to_insert = np.concatenate((states, states), axis=1)
elif SorNS == 'NS':
    states_to_insert = np.concatenate((next_states, next_states), axis=1)

start = time.time()
p = index.Property()
p.dat_extension = 'data'
p.idx_extension = 'index'
p.dimension = state_dim
idx = index.Index(interleaved=True, properties = p)
for i in range(len(states)):
    idx.insert(i, states_to_insert[i])
end = time.time()
print(end-start)

start = time.time()
reach_nst = [[]]*len(states)
for i in range(len(states)):
    check = np.concatenate((mins[i], maxes[i]))
    reach = list(idx.intersection(check))
    reach2 = np.array(reach)
    if SorNS == 'S':
        nst_test = states[reach2].to(device)
    elif SorNS == 'NS':
        nst_test = next_states[reach2].to(device)
    st_test = torch.repeat_interleave(states[i].reshape(1,state_dim),len(nst_test),0).to(device)
    act_pred0 = inv_ss[0](st_test, nst_test).detach().cpu().numpy()
    act_pred1 = inv_ss[1](st_test, nst_test).detach().cpu().numpy()
    act_pred2 = inv_ss[2](st_test, nst_test).detach().cpu().numpy()
    act = np.mean((act_pred0, act_pred1, act_pred2), axis=0)
    act = torch.Tensor(act).to(device)

    delta_s0 = f_sa0(st_test, act).detach().cpu().numpy()
    delta_s1 = f_sa1(st_test, act).detach().cpu().numpy()
    delta_s2 = f_sa2(st_test, act).detach().cpu().numpy()
    delta_s3 = f_sa3(st_test, act).detach().cpu().numpy()
    delta_s4 = f_sa4(st_test, act).detach().cpu().numpy()
    delta_s5 = f_sa5(st_test, act).detach().cpu().numpy()
    delta_s6 = f_sa6(st_test, act).detach().cpu().numpy()

    delta_s = torch.Tensor(np.mean((delta_s0, delta_s1, delta_s2, delta_s3, delta_s4, delta_s5, delta_s6),
                                   axis=0)).to(device)
    diff_minmax = maxes[i] - mins[i]
    nst_model = (st_test + delta_s).detach().cpu().numpy()
    if norm == 'linf':
        nst_error = (np.abs(nst_model - nst_test.detach().cpu().numpy()) / diff_minmax).max(1) 
    elif norm == 'l1':
        nst_error = np.abs(np.abs(nst_model - nst_test.detach().cpu().numpy())/ diff_minmax).sum(1)
    elif norm == 'l2':
        nst_error = (((nst_model - nst_test.detach().cpu().numpy())/ diff_minmax)**2).sum(1)
    elif norm == 'none':
        nst_error = np.zeros((nst_model.shape[0],))
    #### Every state should have at least the next state as reachable
    if SorNS == 'S':
        indd = i+1
    elif SorNS == 'NS':
        indd = i

    indices = np.array((nst_error <= delta_X)).nonzero()[0]  
    if len(indices) ==0:
        reachable_set = [indd]
    elif len(indices) == 1:
        reachable_set = [reach2[int(indices)]]
    else:
        reachable_set = reach2[indices]

    if len(np.where(reachable_set == indd)) == 0:
        reachable_set.append(indd)
    reach_nst[i] = reachable_set

    if i% 10000 == 0:
        end = time.time()
        print("Iteration", i,
              "time taken = ", (end-start)// 3600, "hrs", ((end-start)%3600)// 60 , "mins and ", (end-start) %60, "secs")

with open(f"Reachability_indexes/Ablations/Reachable_{norm}_{delta_X}_{env_name}_{SorNS}.pkl", 'wb') as f:
    pickle.dump(reach_nst, f)
idx.close()

lens = np.array(list(map(len, reach_nst)))
