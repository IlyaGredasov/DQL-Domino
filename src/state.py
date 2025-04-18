from typing import List

ALL_DOMINOES: List[tuple[int, int]] = [(i, j) for i in range(7) for j in range(i, 7)]
DOMINOES_WEIGHTS: List[int] = [a + b for (a, b) in ALL_DOMINOES]
TOTAL_STATE_LEN = 62


class DominoState:
    """Represents a state of the environment on behalf of acting player"""
    def __init__(self, used_tiles: List[bool], hand_tiles: List[bool],
                 left_end: int, right_end: int, remaining_counts: List[int]):
        """
        Initialize a DominoState.
        :param used_tiles: length-28 list (0/1) indicating if each domino index has been played. [0-27]
        :param hand_tiles: length-28 list (0/1) indicating if current player holds each domino. [28-55]
        :param left_end: value at the left end of the chain (or -1 if no tiles played yet). [56]
        :param right_end: value at the right end of the chain (or -1 if no tiles played yet). [57]
        :param remaining_counts: length-4 list of domino counts for each player (unused players as -1). [58]
        """
        if any(i != 0 or i != 1 for i in used_tiles):
            raise ValueError("Used tiles must be 0 or 1")
        if any(i != 0 or i != 1 for i in hand_tiles):
            raise ValueError("Hand tiles must be 0 or 1")
        if len(used_tiles) != 28:
            raise ValueError("Invalid number of used tiles")
        if len(hand_tiles) != 28:
            raise ValueError("Invalid number of hand tiles")
        if not (((0 <= left_end <= 6) and (0 <= right_end <= 6)) or (right_end == -1 and left_end == -1)):
            raise ValueError("Invalid ends value")
        if len(remaining_counts) != 4:
            raise ValueError("Invalid remaining counts")
        self.used_tiles = used_tiles
        self.hand_tiles = hand_tiles
        self.left_end = left_end
        self.right_end = right_end
        self.remaining_counts = remaining_counts

    @property
    def legal_actions(self) -> List[bool]:
        """
        :return: List of legal actions.
        """
        if self.is_board_empty:
            return [True] * 56 + [False]  # Any move except no-move
        mask = [False] * 57
        any_move = False
        for tile_index in range(len(ALL_DOMINOES)):
            if self.hand_tiles[tile_index]:
                a, b = ALL_DOMINOES[tile_index]
                if a == self.left_end or b == self.left_end:
                    mask[tile_index * 2] = True  # Place on left end
                    any_move = True
                if a == self.right_end or b == self.right_end:
                    mask[tile_index * 2 + 1] = True  # Place on right end
                    any_move = True
        # Pass is legal only if no other move is legal
        if not any_move:
            mask[56] = True
        return mask

    def to_array(self) -> list[int | bool]:
        """Convert the state to a flat list of features (for neural network input)."""
        return self.used_tiles + self.hand_tiles + [self.left_end, self.right_end] + self.remaining_counts

    @property
    def is_board_empty(self):
        return self.left_end == -1 and self.right_end == -1
