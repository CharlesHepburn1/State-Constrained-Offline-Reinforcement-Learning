import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data


class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(InverseModel, self).__init__()

        self.l1 = nn.Linear(2*state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, next_state):
        a = torch.cat([state, next_state], 1)
        action = F.relu(self.l1(a))
        action = F.relu(self.l2(action))
        action = F.relu(self.l3(action))
        return self.l4(action)


class DetInvDynamics(object):
    def __init__(self, state_dim, action_dim, lr=3e-4, hidden_dim = 256,device="cpu"):

        self.InverseModel = InverseModel(state_dim, action_dim, hidden_dim = hidden_dim).to(device)
        self.inverse_optimizer = torch.optim.Adam(self.InverseModel.parameters(), lr=lr)

        self.device = device

    def train_dynamics(self, replay_buffer_train, replay_buffer_val):
        # Training #
        training_loss = 0
        for i, batch in enumerate(replay_buffer_train):
            state_batch = batch[0].float().to(self.device)
            action_batch = batch[1].float().to(self.device)
            next_state_batch = batch[3].float().to(self.device)

            action_pred = self.InverseModel(state_batch, next_state_batch)
            train_loss = F.mse_loss(action_pred, action_batch)

            self.inverse_optimizer.zero_grad()
            train_loss.backward()
            self.inverse_optimizer.step()

            training_loss += train_loss.item()

        # Validation #
        validation_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(replay_buffer_val):
                state_batch = batch[0].float().to(self.device)
                action_batch = batch[1].float().to(self.device)
                next_state_batch = batch[3].float().to(self.device)

                action_pred = self.InverseModel(state_batch, next_state_batch)
                val_loss = F.mse_loss(action_pred, action_batch)

                validation_loss += val_loss.item()

        training_loss /= len(replay_buffer_train)
        validation_loss /= len(replay_buffer_val)

        return training_loss, validation_loss
