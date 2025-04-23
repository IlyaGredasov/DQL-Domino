import random
from typing import Set, List, Optional

from src.action import DominoAction
from src.state import DominoState, ALL_DOMINOES

REWARD_SCALE = 12.0


class DominoEnvironment:
    """Environment for the Dominoes game (2-4 players) following standard Domino rules."""

    def __init__(self, num_players: int = 2, agent_indices: List[int] = None):
        if agent_indices is None:
            agent_indices = {0}
        if not 2 <= num_players <= 4:
            raise ValueError("Number of players must be between 2 and 4")
        self.num_players = num_players
        self.player_hands: List[Set[int]] = []
        self.used_tiles: Set[int] = set()
        self.left_end: int = -1
        self.right_end: int = -1
        self.current_player: int = 0
        self.consecutive_passes: int = 0
        self.draw_pile: List[int] = []
        self.agent_indices: List[int] = agent_indices
        self.final_rewards: Optional[dict[int, float]] = None

    @property
    def current_state(self) -> DominoState:
        used_flags = [i in self.used_tiles for i in range(28)]
        hand_flags = [i in self.player_hands[self.current_player] for i in range(28)]
        remaining_counts = [
            len(self.player_hands[i]) if i < self.num_players else -1
            for i in range(4)
        ]
        return DominoState(used_flags, hand_flags, self.left_end, self.right_end,
                           remaining_counts)

    def reset(self) -> DominoState:
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
        return self.current_state

    def apply_action(self, action: DominoAction):
        if action.is_pass:
            if self.current_state.is_board_empty:
                raise ValueError("Can't apply pass action on empty board")
            self.consecutive_passes += 1
            return
        tile_idx = action.tile_index
        a, b = ALL_DOMINOES[tile_idx]
        if self.current_state.is_board_empty:
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
        self.player_hands[self.current_player].remove(tile_idx)
        self.used_tiles.add(tile_idx)
        self.consecutive_passes = 0
        if len(self.player_hands[self.current_player]) == 0:
            # Mark a special flag to indicate game end due to win
            self.consecutive_passes = self.num_players  # force game end condition

    def draw_tile(self, tile_index: Optional[int] = None) -> tuple[bool, int]:
        if not self.draw_pile:
            return False, -1

        if tile_index is None:
            tile = self.draw_pile.pop()
        elif 0 <= tile_index < len(self.draw_pile):
            tile = self.draw_pile.pop(tile_index)
        else:
            return False, -1

        self.player_hands[self.current_player].add(tile)
        return True, tile

    def check_terminal_state(self) -> Optional[dict[int, float]]:
        for p in range(self.num_players):
            if len(self.player_hands[p]) == 0:
                self.final_rewards = self.finalize_game(winner=p)
                return self.final_rewards

        if self.consecutive_passes >= self.num_players and not self.draw_pile:
            self.final_rewards = self.finalize_game(winner=None)
            return self.final_rewards

        return None

    def step(self, action: DominoAction) -> tuple[DominoState, float, bool]:
        if not self.current_state.legal_actions[action.index]:
            raise ValueError(f"Illegal action {action.index} for current state.")

        if action.is_pass:
            step_reward = -25.0 / REWARD_SCALE  # reward scaling to not distract agent too much
        else:
            tile_idx = action.tile_index
            a, b = ALL_DOMINOES[tile_idx]
            step_reward = (a + b) / REWARD_SCALE  # reward scaling to not distract agent too much

        self.apply_action(action)

        end_rewards = self.check_terminal_state()
        if end_rewards is not None:
            total_reward = step_reward + end_rewards.get(self.current_player, 0.0)
            return self.current_state, total_reward, True

        self.current_player = (self.current_player + 1) % self.num_players
        return self.current_state, step_reward, False

    def finalize_game(self, winner: Optional[int]) -> dict[int, float]:
        def pip_sum(tiles: set[int]) -> int:
            return sum(ALL_DOMINOES[i][0] + ALL_DOMINOES[i][1] for i in tiles)

        rewards: dict[int, float] = {}

        if winner is None:
            pip_counts = [pip_sum(self.player_hands[p]) for p in range(self.num_players)]
            avg_pips = sum(pip_counts) / self.num_players
            for p in range(self.num_players):
                rewards[p] = avg_pips - pip_counts[p]  # (p_1+...+p_n)/n-p_i
        else:
            for p in range(self.num_players):
                if p == winner:
                    rewards[p] = float(
                        sum(pip_sum(self.player_hands[opp]) for opp in range(self.num_players) if opp != p)
                    )  # p_1+...p_n (all pips except self == 0)
                else:
                    rewards[p] = -float(pip_sum(self.player_hands[p]))  # -p_i
        for p, hand in enumerate(self.player_hands):
            if hand == {0}:  # special for {(0, 0)}
                rewards[p] = -10.0
        return rewards

    def __str__(self) -> str:
        board = f"Board: Left={self.left_end}, Right={self.right_end}, Pile={len(self.draw_pile)} tiles"
        players = []
        for i, hand in enumerate(self.player_hands):
            dominos = ', '.join(str(ALL_DOMINOES[d]) for d in sorted(hand))
            players.append(f"P{i} ({len(hand)} tiles): {dominos}")
        return (
                "[DominoEnvironment]\n"
                f"{board}\n"
                f"Current turn: Player {self.current_player}\n"
                + "\n".join(players)
        )

    def __repr__(self):
        return str(self)
