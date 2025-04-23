from typing import List

ALL_DOMINOES: List[tuple[int, int]] = [(i, j) for i in range(7) for j in range(i, 7)]
DOMINOES_WEIGHTS: List[int] = [a + b for (a, b) in ALL_DOMINOES]
TOTAL_STATE_LEN = 62


class DominoState:
    """Represents a state of the environment on behalf of acting player."""

    def __init__(self, used_tiles: List[bool], hand_tiles: List[bool],
                 left_end: int, right_end: int, remaining_counts: List[int]):
        if any(i != 0 and i != 1 for i in used_tiles):
            raise ValueError("Used tiles must be 0 or 1")
        if any(i != 0 and i != 1 for i in hand_tiles):
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
        mask = [False] * 57
        if self.is_board_empty:
            for tile_idx, held in enumerate(self.hand_tiles):
                if held:
                    mask[tile_idx * 2] = True   # Place on left end
                    mask[tile_idx * 2 + 1] = True  # Place on right end
            return mask
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
        return self.used_tiles + self.hand_tiles + [self.left_end, self.right_end] + self.remaining_counts

    @property
    def is_board_empty(self):
        return self.left_end == -1 and self.right_end == -1

    def __str__(self):
        board_info = f"Board ends: Left={self.left_end}, Right={self.right_end}"
        used = [i for i, used in enumerate(self.used_tiles) if used]
        hand = [i for i, held in enumerate(self.hand_tiles) if held]
        used_dominoes = ', '.join(str(ALL_DOMINOES[i]) for i in used)
        hand_dominoes = ', '.join(str(ALL_DOMINOES[i]) for i in hand)
        rem = ', '.join(f"P{i}: {c}" if c != -1 else f"P{i}: -" for i, c in enumerate(self.remaining_counts))
        return (
            f"[DominoState]\n"
            f"{board_info}\n"
            f"Used: {used_dominoes or 'None'}\n"
            f"Hand: {hand_dominoes or 'None'}\n"
            f"Remaining: {rem}"
        )

    def __repr__(self):
        return str(self)
