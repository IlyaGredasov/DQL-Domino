import torch
import torch.nn as nn
from torch import optim

from src.action import TOTAL_ACTIONS
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
