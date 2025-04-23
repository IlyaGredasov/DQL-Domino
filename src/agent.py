import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.action import TOTAL_ACTIONS, DominoAction
from src.state import TOTAL_STATE_LEN


class DQLNetwork(nn.Module):
    """DQL neural network for agent"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 384),
            nn.LayerNorm(384),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),

            nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DominoAgent:
    """Deep Q-Learning agent for domino"""

    def __init__(self, state_dim: int = TOTAL_STATE_LEN, action_dim: int = TOTAL_ACTIONS, gamma: float = 0.995,
                 lr: float = 2e-5, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9994):
        self.state_dim = state_dim  # 62
        self.action_dim = action_dim  # 57
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQLNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQLNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def policy_net_forward(self, state_array: List[int]) -> torch.Tensor:
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)  # shape [1, state_dim]
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
        return q_values

    def select_action(self, state, legal_actions: list[int], training: bool = True, suggestions: bool = False) -> int:
        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_np = np.array(state.to_array(), dtype=np.float32)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        q_values = None
        if training:
            q_values = self.policy_net(state_tensor).squeeze()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze()

        mask = torch.full_like(q_values, float('-inf'))
        for i in legal_actions:
            mask[i] = q_values[i]
        probs = F.softmax(mask, dim=0).detach().cpu().numpy()

        if suggestions:
            print("Agent's softmax probabilities (legal actions only):")
            for i in sorted(legal_actions, key=lambda x: -probs[x]):
                print(f"[{i}] {DominoAction(i)}: P={probs[i]:.4f}")

        return int(torch.argmax(mask).item())

    def train_step(self, batch: dict) -> float:
        state_batch = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch['actions'], dtype=torch.int64, device=self.device)
        reward_batch = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch['dones'], dtype=torch.bool, device=self.device)

        # Q(s, a)
        q_values = self.policy_net(state_batch)
        q_values_taken = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Q(s', a') for Double DQN
        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_batch)
            next_q_target = self.target_net(next_state_batch)
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            target = reward_batch + self.gamma * next_q_values * (~done_batch)

        loss = F.smooth_l1_loss(q_values_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
