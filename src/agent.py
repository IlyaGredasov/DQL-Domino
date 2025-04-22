import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.action import TOTAL_ACTIONS
from src.state import TOTAL_STATE_LEN, DominoState


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

    def __init__(
            self,
            state_dim: int = TOTAL_STATE_LEN,
            action_dim: int = TOTAL_ACTIONS,
            gamma: float = 0.995,
            lr: float = 5e-4,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: float = 0.995,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQLNetwork(state_dim, action_dim)
        self.target_net = DQLNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def policy_net_forward(self, state_array: List[int]) -> torch.Tensor:
        """
        Get Q-values from the policy network for a given state (detached, for inference or self-play).
        :param state_array: List[int] representing flattened DominoState.
        :return: torch.Tensor of shape [action_dim]
        """
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)  # shape [1, state_dim]
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
        return q_values

    def select_action(self, state: DominoState, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy with masking.
        :param state: DominoState object (must implement .legal_actions).
        :param training: Whether to apply epsilon-greedy exploration.
        :return: Chosen action index (0–56)
        """
        mask = state.legal_actions
        legal_actions = [i for i, valid in enumerate(mask) if valid]

        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Greedy choice
        state_tensor = torch.tensor(state.to_array(), dtype=torch.float32).unsqueeze(0)
        q_values = self.policy_net(state_tensor)[0]

        # Mask out illegal actions
        q_values[~torch.tensor(mask)] = -1e9

        return int(torch.argmax(q_values).item())

    def train_step(self, batch: dict) -> float:
        """
        Perform one training step using a batch of experiences.
        :param batch: Dict with keys 'states', 'actions', 'rewards', 'next_states', 'dones'.
        :return: The computed loss value.
        """
        # Convert batch elements to tensors
        state_batch = torch.tensor(batch['states'], dtype=torch.float32)  # [B, state_dim]
        action_batch = torch.tensor(batch['actions'], dtype=torch.int64)  # [B]
        reward_batch = torch.tensor(batch['rewards'], dtype=torch.float32)  # [B]
        next_state_batch = torch.tensor(batch['next_states'], dtype=torch.float32)  # [B, state_dim]
        done_batch = torch.tensor(batch['dones'], dtype=torch.bool)  # [B]

        # Q(s, a) — current predicted Q values for taken actions
        q_values = self.policy_net(state_batch)  # [B, action_dim]
        q_values_taken = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # [B]

        # Double DQN target: max_a' Q_target(s', argmax_a Q_policy(s', a))
        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_batch)  # [B, action_dim]
            next_q_target = self.target_net(next_state_batch)  # [B, action_dim]
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)  # [B, 1]
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)  # [B]
            target = reward_batch + self.gamma * next_q_values * (~done_batch)

        # Loss calculation
        loss = F.smooth_l1_loss(q_values_taken, target)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy the weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
