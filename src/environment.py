import random
from typing import Set

import numpy as np
from numpy.typing import NDArray

from src.action import DominoAction
from src.state import DominoState, ALL_DOMINOES, DOMINOES_WEIGHTS


class DominoEnvironment:
    """Environment for the Dominoes game (2-4 players) following standard Block Domino rules."""

    def __init__(self, num_players: np.int_ = 2, agent_indices: NDArray[np.int_] = (0,)):
        if agent_indices is None:
            agent_indices = {0}
        if not 2 <= num_players <= 4:
            raise ValueError("Number of players must be between 2 and 4")
        self.num_players = num_players
        self.player_hands: NDArray[Set[np.int_]] = np.array([], dtype=object)
        self.used_tiles: Set[np.int_] = set()
        self.left_end: np.int_ = np.int_(-1)
        self.right_end: np.int_ = np.int_(-1)
        self.current_player: np.int_ = np.int_(0)
        self.consecutive_passes: np.int_ = np.int_(0)
        self.draw_pile: NDArray[np.int_] = np.array([], dtype=np.int_)
        self.opponent_policy = None  # For future use
        self.agent_indices: Set[np.int_] = {np.int_(x) for x in agent_indices}

    @property
    def is_board_empty(self) -> bool:
        return self.left_end == -1 and self.right_end == -1

    def get_player_state(self, player_index: np.int_) -> DominoState:
        used_flags = np.array([1 if i in self.used_tiles else 0 for i in range(28)])
        hand_flags = np.array([1 if i in self.player_hands[player_index] else 0 for i in range(28)])
        remaining_counts = np.array([
            len(self.player_hands[i]) if i < self.num_players else -1
            for i in range(4)
        ])
        return DominoState(used_flags, hand_flags, self.left_end, self.right_end, remaining_counts)

    def reset(self) -> DominoState:
        """Start a new domino game. Deal tiles and return initial state for player 0."""
        # Create a shuffled domino set
        tiles = list(range(len(ALL_DOMINOES)))  # [0-27]
        random.shuffle(tiles)
        tiles_per_player = 7
        self.player_hands = []
        start_index = 0
        for _ in range(self.num_players):
            self.player_hands.append(set(tiles[start_index:start_index + 7]))
            start_index += 7

        self.draw_pile = tiles[start_index:]

        self.used_tiles = set()
        self.left_end = -1
        self.right_end = -1
        self.current_player = 0  # maybe randint(0,3)
        self.consecutive_passes = 0
        return self.get_player_state(np.int_(self.current_player))



