import random
from typing import Set, List, Optional

from src.action import DominoAction
from src.state import DominoState, ALL_DOMINOES


class DominoEnvironment:
    """Environment for the Dominoes game (2-4 players) following standard Block Domino rules."""

    def __init__(self, num_players: int = 2, agent_indices: List[int] = (0,)):
        if agent_indices is None:
            agent_indices = {0}
        if not 2 <= num_players <= 4:
            raise ValueError("Number of players must be between 2 and 4")
        self.num_players = num_players
        self.player_hands: List[Set[int]] = []
        self.used_tiles: Set[int] = set()
        self.left_end: int = int(-1)
        self.right_end: int = int(-1)
        self.current_player: int = int(0)
        self.consecutive_passes: int = int(0)
        self.draw_pile: List[int] = []
        self.opponent_policy = None  # For future use
        self.agent_indices: Set[int] = {int(x) for x in agent_indices}

    @property
    def is_board_empty(self) -> bool:
        return self.left_end == -1 and self.right_end == -1

    def get_player_state(self, player_index: int) -> DominoState:
        used_flags = [bool(1 if i in self.used_tiles else 0) for i in range(28)]
        hand_flags = [bool(1 if i in self.player_hands[player_index] else 0) for i in range(28)]
        remaining_counts = [
            len(self.player_hands[i]) if i < self.num_players else -1
            for i in range(4)
        ]
        return DominoState(used_flags, hand_flags, self.left_end, self.right_end, remaining_counts)

    def reset(self) -> DominoState:
        """Start a new domino game. Deal tiles and return initial state for player 0."""
        tiles = list(range(len(ALL_DOMINOES)))  # [0-27]
        random.shuffle(tiles)
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
        return self.get_player_state(int(self.current_player))

    def apply_action(self, player: int, action: DominoAction):
        """Apply the given action for the specified player (updates game state)."""
        if action.is_pass:
            if self.is_board_empty:
                raise ValueError("Can't apply pass action on empty board")
            self.consecutive_passes += 1
            return
        tile_idx = action.tile_index
        a, b = ALL_DOMINOES[tile_idx]
        if self.is_board_empty:
            # First tile played
            if action.play_on_left:
                self.left_end, self.right_end = a, b
            else:
                self.left_end, self.right_end = b, a
        elif action.play_on_left:
            # Playing on left end
            if self.left_end == a:
                self.left_end = b
            elif self.left_end == b:
                self.left_end = a
            else:
                raise RuntimeError(
                    f"Illegal move: tile {ALL_DOMINOES[tile_idx]} cannot be played on left_end {self.left_end}")
        else:
            # Playing on right end
            if self.right_end == a:
                self.right_end = b
            elif self.right_end == b:
                self.right_end = a
            else:
                raise RuntimeError(
                    f"Illegal move: tile {ALL_DOMINOES[tile_idx]} cannot be played on right_end {self.right_end}")
        self.player_hands[player].remove(tile_idx)
        self.used_tiles.add(tile_idx)
        self.consecutive_passes = 0
        # If a player empties their hand by playing a tile, they win
        if len(self.player_hands[player]) == 0:
            # Mark a special flag to indicate game end due to win
            self.consecutive_passes = self.num_players  # force game end condition

    def apply_random_action(self):
        state = self.get_player_state(self.current_player)
        actions = state.legal_actions
        self.apply_action(self.current_player, random.choice(actions))

    def step(self, action: DominoAction, agent_indices: set[int]) -> tuple[DominoState, dict[int, float], bool]:
        """
        Perform a full game cycle starting from the current player's action.
        :param action: The action chosen by the current player (must be agent-controlled).
        :param agent_indices: Indices of agent-controlled players.
        :return: (next_agent_state, reward_dict, done)
        """
        self.apply_action(self.current_player, action)

        if len(self.player_hands[self.current_player]) == 0:
            rewards = self.finalize_game(winner=self.current_player, agent_indices=agent_indices)
            return self.get_player_state(self.current_player), rewards, True

        for _ in range(1, self.num_players):
            self.current_player = (self.current_player + 1) % self.num_players

            while True:
                state = self.get_player_state(self.current_player)
                legal_mask = state.legal_actions

                if any(legal_mask[:56]):
                    if self.current_player in agent_indices:
                        # It's the next agent's turn
                        return state, {agent: 0.0 for agent in agent_indices}, False
                    else:
                        self.apply_random_action()
                        break
                else:
                    if self.draw_pile:
                        drawn_tile = self.draw_pile.pop()
                        self.player_hands[self.current_player].add(drawn_tile)
                        continue  # try again
                    else:
                        self.apply_action(self.current_player, DominoAction(56))
                        break

            if len(self.player_hands[self.current_player]) == 0:
                rewards = self.finalize_game(winner=self.current_player, agent_indices=agent_indices)
                return self.get_player_state(self.current_player), rewards, True
            if self.consecutive_passes >= self.num_players and not self.draw_pile:
                rewards = self.finalize_game(winner=None, agent_indices=agent_indices)
                return self.get_player_state(self.current_player), rewards, True

        self.current_player = (self.current_player + 1) % self.num_players
        next_state = self.get_player_state(self.current_player)
        return next_state, {agent: 0.0 for agent in agent_indices}, False

    def finalize_game(self, winner: Optional[int], agent_indices: set[int]) -> dict[int, float]:
        """
        Calculate rewards for all agent-controlled players.
        - Win: reward = sum of opponent pips
        - Blocked draw: reward = average_pips - agent_pips
        """

        def pip_sum(tiles):
            return sum(ALL_DOMINOES[i][0] + ALL_DOMINOES[i][1] for i in tiles)

        rewards = {}
        if winner is None:
            # Blocked game: lowest pip count wins, but no actual winner
            pip_counts = [pip_sum(self.player_hands[p]) for p in range(self.num_players)]
            avg_pips = sum(pip_counts) / self.num_players
            for agent in agent_indices:
                agent_pips = pip_counts[agent]
                rewards[agent] = avg_pips - agent_pips
        else:
            for agent in agent_indices:
                if agent == winner:
                    rewards[agent] = float(
                        sum(pip_sum(self.player_hands[p]) for p in range(self.num_players) if p != agent)
                    )
                else:
                    rewards[agent] = -float(pip_sum(self.player_hands[agent]))
        return rewards
