from src.state import ALL_DOMINOES
TOTAL_ACTIONS = 57


class DominoAction:
    """Represents an action (playing a specific domino on left/right, or passing)."""

    def __init__(self, action_index: int):
        if action_index < 0 or action_index > 56:
            raise ValueError("Invalid action index")
        self.index: int = action_index
        self.is_pass: bool = (action_index == 56)
        if not self.is_pass:
            self.tile_index: int = action_index // 2
            self.play_on_left: bool = (action_index % 2 == 0)

    def __str__(self):
        if self.is_pass:
            return "Pass"
        tile = ALL_DOMINOES[self.tile_index]
        side = "Left" if self.play_on_left else "Right"
        return f"Play {tile} on {side}"

    def __repr__(self):
        return str(self)
