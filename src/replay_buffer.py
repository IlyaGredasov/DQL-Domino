import random
import numpy as np
from collections import deque
from typing import Deque, Tuple


class ReplayBuffer:
    """Replay buffer for NN."""

    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool,
             action_mask: np.ndarray, next_action_mask: np.ndarray):
        self.buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = zip(*batch)

        return (
            np.stack(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.stack(next_state),
            np.array(done, dtype=np.float32),
            np.stack(mask),
            np.stack(next_mask),
        )

    def __len__(self) -> int:
        return len(self.buffer)
