import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data

# Define forward dynamics model

class Dynamics2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Dynamics2, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        ns = F.relu(self.l1(torch.cat([state, action], dim=-1)))
        ns = F.relu(self.l2(ns))
        ns = F.relu(self.l3(ns))
        ns = F.relu(self.l4(ns))
        ns = self.l5(ns)

        return ns

class DetDynamics(object):
    def __init__(self, state_dim, action_dim, lr=3e-4, device="cpu"):

        self.dynamics = Dynamics2(state_dim, action_dim).to(device)  ## StaCQ code uses Dynamics (state_dim, action_dim).to(device)
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=lr)

        self.device = device

    def train_dynamics(self, replay_buffer_train, replay_buffer_val):
        # Training #
        training_loss = 0
        for i, batch in enumerate(replay_buffer_train):
            state_batch = batch[0].float().to(self.device)
            action_batch = batch[1].float().to(self.device)
            next_state_batch = batch[3].float().to(self.device)

            next_state_pred = self.dynamics(state_batch, action_batch)
            delta_s = next_state_batch - state_batch
            train_loss = F.mse_loss(next_state_pred, delta_s)
            # train_loss = F.mse_loss(next_state_pred, next_state_batch)

            self.dynamics_optimizer.zero_grad()
            train_loss.backward()
            self.dynamics_optimizer.step()

            training_loss += train_loss.item()

        # Validation #
        validation_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(replay_buffer_val):
                state_batch = batch[0].float().to(self.device)
                action_batch = batch[1].float().to(self.device)
                next_state_batch = batch[3].float().to(self.device)

                next_state_pred = self.dynamics(state_batch, action_batch)
                delta_s = next_state_batch - state_batch
                val_loss = F.mse_loss(next_state_pred, delta_s)
                # val_loss = F.mse_loss(next_state_pred, next_state_batch)

                validation_loss += val_loss.item()

        training_loss /= len(replay_buffer_train)
        validation_loss /= len(replay_buffer_val)

        return training_loss, validation_loss
