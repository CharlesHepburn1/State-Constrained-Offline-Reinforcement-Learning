# Imports
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import time
import itertools
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

class Reward_MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Reward_MLP, self).__init__()
        self.input_dim =  2*state_dim
        self.l1 = nn.Linear(self.input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
    def forward(self, state, next_state):
        p = F.relu(self.l1(torch.cat([state, next_state], dim=-1)))
        p = F.relu(self.l2(p))
        p = self.l3(p)
        return p

# Define critic network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.l4 = nn.Linear(state_dim + state_dim, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, 1)

        self.l7 = nn.Linear(state_dim + state_dim, hidden_size)
        self.l8 = nn.Linear(hidden_size, hidden_size)
        self.l9 = nn.Linear(hidden_size, 1)

        self.l10 = nn.Linear(state_dim + state_dim, hidden_size)
        self.l11 = nn.Linear(hidden_size, hidden_size)
        self.l12 = nn.Linear(hidden_size, 1)

    def forward(self, state, next_state):
        q1 = F.relu(self.l1(torch.cat([state, next_state], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, next_state], dim=-1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        q3 = F.relu(self.l7(torch.cat([state, next_state], dim=-1)))
        q3 = F.relu(self.l8(q3))
        q3 = self.l9(q3)

        q4 = F.relu(self.l10(torch.cat([state, next_state], dim=-1)))
        q4 = F.relu(self.l11(q4))
        q4 = self.l12(q4)

        return torch.squeeze(q1, dim=-1), torch.squeeze(q2, dim=-1), torch.squeeze(q3, dim=-1), torch.squeeze(q4,
                                                                                                              dim=-1)

    def Q1(self, state, next_state):
        q1 = F.relu(self.l1(torch.cat([state, next_state], dim=-1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return torch.squeeze(q1, dim=-1)


# Define actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))

        return a


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, normal_variance=0.1, batch_size=256, gamma=0.99, tau=0.005,
                 lr=3e-4, policy_noise=0.2,policy_noise_act = 0.2, noise_clip=0.5,noise_clip_act =0.5, policy_freq=1, lmbda=0.5,
                 Actor_funct = 'StateBCReg', Ens_size = 2, Actor_ens_min = False, ind_targ = False, device="cpu"):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, int(1e6))

        self.critic = Critic(state_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.Rew = Reward_MLP(state_dim, hidden_dim=256).to(device)
        self.rew_optimizer = torch.optim.Adam(self.Rew.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_noise_act = policy_noise_act
        self.noise_clip = noise_clip
        self.noise_clip_act = noise_clip_act
        self.policy_freq = policy_freq
        self.lmbda = lmbda
        self.device = device
        self.batch_size = batch_size
        self.buffer_length = 0
        self.norm_var = normal_variance
        self.Ens_size = Ens_size
        self.Actor_ens_min = Actor_ens_min

        self.Actor_funct = Actor_funct
        self.independent_targets = ind_targ
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.lmbda_history = []

        self.total_it = 0

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def remove_for_memory(self, array, leng):
        if leng <=500 :
            return array
        else:
            return np.random.choice(array, 500, replace = False)
    def retrieve_from_buffer1(self, array, states):
        array_tens = torch.from_numpy(np.array(array)).to(torch.int)  # replay_buffer[0] is states, replay_buffer[3] is next_states
        return torch.index_select(states, 0,array_tens)  # torch.index_select(replay_buffer[0], 0, array_tens)#
    def retrieve_from_buffer2(self, array, states): #sometime I don't want it to pick the last state
        index_end = np.where(np.array(array) == [self.buffer_length])[0]
        if len(index_end) >0:
            array[index_end] = [1]

        array_tens = torch.from_numpy(np.array(array)).to(torch.int)  # replay_buffer[0] is states, replay_buffer[3] is next_states

        return torch.index_select(states, 0,array_tens)  # torch.index_select(replay_buffer[0], 0, array_tens)#

    def add_1(self, array):
        return np.array(array)+1

    def retrieve_ind(self, ind, list):
        indd = list[ind[0]]
        if indd == self.buffer_length:
            indd -=1

        return indd


    def train2(self, replay_buffer, list_idx, forward_model, inverse_model, iterations=1):
        self.buffer_length = len(replay_buffer[0])
        for it in range(iterations):
            self.total_it += 1
            indices = torch.randint(0, len(replay_buffer[0]), size=(self.batch_size,))
            indices_np = indices.numpy()
            state1 = torch.index_select(replay_buffer[0], 0, indices).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, indices).to(self.device)
            reward1 = torch.index_select(replay_buffer[2], 0, indices).to(self.device)
            next_state1 = torch.index_select(replay_buffer[3], 0, indices).to(self.device)
            done = torch.index_select(replay_buffer[4], 0, indices).to(self.device)

            ###############################################################################################
            ### Critic - supplement states, next_states and rewards with the ones from the reachability ###

            list_idx_take = np.take(list_idx, indices_np)
            # list_idx_take = np.take(list_idx, indices_np + 1)
            lens = np.array(list(map(len, list_idx_take)))

            if np.min(lens) == 0:
                # These should be the points where the done flag ==True (as always the next state in the trajectory is reachable
                # So I can set the reachable state to be whatever (e.g. state 0) and they will never be evaluated
                idx_mins = np.argwhere(lens == 0)
                for i in range(len(idx_mins)):
                    ind_m = idx_mins[i][0]
                    list_idx_take[ind_m] = [0]  #
                # lens = np.array(list(map(len, list_idx_take)))

            list_idx_take = np.take(list_idx, indices_np)
            lens = np.array(list(map(len, list_idx_take)))

            list_idx_take_merged = torch.Tensor(np.array(list(itertools.chain(*list_idx_take)))).int()
            state = torch.index_select(replay_buffer[0], 0, list_idx_take_merged).to(self.device)
            action = torch.index_select(replay_buffer[1], 0, list_idx_take_merged).to(self.device) ## We're not actually using this action in any updates
            next_state = torch.index_select(replay_buffer[3], 0, list_idx_take_merged).to(self.device)
            reward = torch.index_select(replay_buffer[2], 0, list_idx_take_merged).to(self.device) ## For antmaze the reward only depends on the next state so will be the same
            # reward = self.Rew(state, next_state).reshape(-1,)
            done = torch.index_select(replay_buffer[4], 0, list_idx_take_merged).to(self.device)
            #############################

            # Critic #
            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

                delta_s0_pol = forward_model[0](next_state, next_action)
                delta_s1_pol = forward_model[1](next_state, next_action)
                delta_s2_pol = forward_model[2](next_state, next_action)
                delta_s3_pol = forward_model[3](next_state, next_action)
                delta_s4_pol = forward_model[4](next_state, next_action)
                delta_s5_pol = forward_model[5](next_state, next_action)
                delta_s6_pol = forward_model[6](next_state, next_action)
                delta_s_pol = torch.mean(torch.stack((delta_s0_pol, delta_s1_pol, delta_s2_pol, delta_s3_pol,
                                                      delta_s4_pol, delta_s5_pol, delta_s6_pol), 0), 0)
                next_next_state = next_state + delta_s_pol

                targetQ1, targetQ2, targetQ3, targetQ4 = self.critic_target(next_state, next_next_state)
                if self.Ens_size == 2:
                    if self.independent_targets:
                        targetQ1 = reward + (1 - done) * self.gamma * targetQ1
                        targetQ2 = reward + (1 - done) * self.gamma * targetQ2
                    else:
                        targetQ = reward + (1 - done) * self.gamma * torch.min(targetQ1, targetQ2)
                elif self.Ens_size == 3:
                    if self.independent_targets:
                        targetQ1 = reward + (1 - done) * self.gamma * targetQ1
                        targetQ2 = reward + (1 - done) * self.gamma * targetQ2
                        targetQ3 = reward + (1 - done) * self.gamma * targetQ3
                    else:
                        targetQ = torch.min(targetQ1, targetQ2)
                        targetQ = reward + (1 - done) * self.gamma * torch.min(targetQ, targetQ3)
                elif self.Ens_size == 4:
                    if self.independent_targets:
                        targetQ1 = reward + (1 - done) * self.gamma * targetQ1
                        targetQ2 = reward + (1 - done) * self.gamma * targetQ2
                        targetQ3 = reward + (1 - done) * self.gamma * targetQ3
                        targetQ4 = reward + (1 - done) * self.gamma * targetQ4
                    else:
                        targetQ = torch.min(targetQ1, targetQ2)
                        targetQ = torch.min(targetQ, targetQ3)
                        targetQ = reward + (1 - done) * self.gamma * torch.min(targetQ, targetQ4)


            currentQ1, currentQ2, currentQ3, currentQ4 = self.critic(state, next_state)
            if self.independent_targets:
                if self.Ens_size ==2:
                    critic_loss = F.mse_loss(currentQ1, targetQ1) + F.mse_loss(currentQ2, targetQ2)
                elif self.Ens_size ==3:
                    critic_loss = F.mse_loss(currentQ1, targetQ1) + F.mse_loss(currentQ2, targetQ2) + \
                                  F.mse_loss(currentQ3, targetQ3)
                elif self.Ens_size ==4:
                    critic_loss = F.mse_loss(currentQ1, targetQ1) + F.mse_loss(currentQ2, targetQ2) +\
                                  F.mse_loss(currentQ3, targetQ3)+ F.mse_loss(currentQ4, targetQ4)
            else:
                if self.Ens_size ==2:
                    critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)
                elif self.Ens_size ==3:
                    critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)+ \
                                  F.mse_loss(currentQ3, targetQ)
                elif self.Ens_size ==4:
                    critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ) + \
                                  F.mse_loss(currentQ3, targetQ)+ F.mse_loss(currentQ4, targetQ)

            self.critic_loss_history.append(critic_loss.item())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor #
            with torch.no_grad():
                nst = torch.index_select(replay_buffer[3], 0, list_idx_take_merged).to(self.device)
                new_done = torch.index_select(replay_buffer[4], 0, list_idx_take_merged).to(self.device)
                state_rep = torch.repeat_interleave(state1, torch.Tensor(lens).to(torch.int).to(self.device), dim=0)

                if self.Actor_ens_min:
                    targ_Q1, targ_Q2, targ_Q3, targ_Q4 = self.critic(state_rep, nst)
                    targ_Q = torch.min(targ_Q1, targ_Q2)
                    if self.Ens_size ==3 or self.Ens_size ==4:
                        targ_Q = torch.min(targ_Q, targ_Q3)
                        if self.Ens_size ==4:
                            targ_Q = torch.min(targ_Q, targ_Q4)
                    targ_Q = targ_Q.detach().cpu().numpy()
                else:
                    targ_Q = (self.critic.Q1(state_rep, nst)).detach().cpu().numpy()
                split_indices = np.cumsum(lens)[:-1]
                sublists = np.split(targ_Q, split_indices)

                ind = np.array(list(map(np.argmax, sublists))).reshape(-1, 1)
                ind_st = np.array(list(map(self.retrieve_ind, ind, list_idx_take)))

                indices_st = torch.from_numpy(ind_st).reshape(-1, )

            best_nst = torch.index_select(replay_buffer[3], 0, indices_st).to(self.device)

            policy_actions = self.actor(state1)


            noise = (torch.randn_like(policy_actions) * self.policy_noise_act).clamp(-self.noise_clip, self.noise_clip)
            policy_actions_ = (policy_actions + noise).clamp(-self.max_action + 1e-4, self.max_action - 1e-4)
            delta_s0_pol = forward_model[0](state1, policy_actions_)
            delta_s1_pol = forward_model[1](state1, policy_actions_)
            delta_s2_pol = forward_model[2](state1, policy_actions_)
            delta_s3_pol = forward_model[3](state1, policy_actions_)
            delta_s4_pol = forward_model[4](state1, policy_actions_)
            delta_s5_pol = forward_model[5](state1, policy_actions_)
            delta_s6_pol = forward_model[6](state1, policy_actions_)
            delta_s_pol = torch.mean(torch.stack((delta_s0_pol, delta_s1_pol, delta_s2_pol, delta_s3_pol, delta_s4_pol,
                                                  delta_s5_pol, delta_s6_pol), 0),0)

            best_action_0 = inverse_model[0](state1, best_nst)
            best_action_1 = inverse_model[1](state1, best_nst)
            best_action_2 = inverse_model[2](state1, best_nst)
            best_action = torch.mean(torch.stack((best_action_0, best_action_1, best_action_2), 0),0)

            # best_action = best_action.clamp(-self.max_action + 1e-4, self.max_action - 1e-4)

            policy_next_state = state1+delta_s_pol

            if self.Actor_ens_min:
                Q1, Q2, Q3, Q4 = self.critic(state1, best_nst)
                Q = torch.min(Q1, Q2)
                if self.Ens_size ==3 or self.Ens_size ==4:
                    Q = torch.min(Q, Q3)
                    if self.Ens_size ==4:
                        Q = torch.min(Q, Q4)
            else:
                Q = self.critic.Q1(state1, best_nst)
            # V = self.critic.Q1(state, policy_next_state)
            with torch.no_grad():
                if self.Actor_ens_min:
                    V1, V2, V3, V4 = self.critic(state1, policy_next_state)
                    V = torch.min(V1, V2)
                    if self.Ens_size == 3 or self.Ens_size == 4:
                        V = torch.min(V, V3)
                        if self.Ens_size == 4:
                            V = torch.min(V, V4)
                else:
                    V = self.critic.Q1(state1, policy_next_state)
            adv = Q - V
            # exp_adv = torch.exp(adv * self.lmbda).clamp(min=None, max=100.0).reshape(-1,1).repeat(1,action.shape[1])
            # actor_loss = (exp_adv * (policy_actions - best_action) ** 2).mean()

            exp_adv = torch.exp(adv * self.lmbda).clamp(min=None, max=100.0).reshape(-1, 1).repeat(1,state.shape[1])

            actor_loss = (exp_adv * (policy_next_state - best_nst) ** 2).mean()

            if self.Actor_ens_min:
                Q1, Q2, Q3, Q4 = self.critic(state1, policy_next_state)
                Q = torch.min(Q1, Q2)
                if self.Ens_size == 3 or self.Ens_size == 4:
                    Q = torch.min(Q, Q3)
                    if self.Ens_size == 4:
                        Q = torch.min(Q, Q4)
            else:
                Q = self.critic.Q1(state1, policy_next_state)
            hyp = self.lmbda / Q.abs().mean().detach()
            actor_loss = -hyp * Q.mean() + F.mse_loss(policy_next_state,best_nst)  # F.mse_loss(policy_actions, best_action)


            self.actor_loss_history.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_scheduler.step()



            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
