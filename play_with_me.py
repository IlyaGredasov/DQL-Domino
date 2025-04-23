from src.agent import DominoAgent
from src.state import DominoState, ALL_DOMINOES

agent = DominoAgent()
agent.load("domino_agent.pt")

used_tiles = [False] * 28
hand_tiles = [False] * 28
left_end = -1
right_end = -1
current_player = 0
num_players = 2
agent_index = 0
remaining = [7 if i < num_players else -1 for i in range(4)]

state = DominoState(used_tiles, hand_tiles, left_end, right_end, remaining)

print("""
Let's play domino, command list:
/start n p - start game with n players (2 - 4), p is your player index (0 to n - 1)
/break - break game
/move a b side - player acts (a, b) onto {side} end
/pass - player passes
/draw n - current player draws n unknown tiles (adjust remaining)
/i_draw a1 b1 a2 b2 ... - you draw known tiles (a1, b1), (a2, b2) ... (add to your hand)
/agent - let agent suggest a move
/show - show current state
/next - go to next player's turn
""")

while True:
    raw = input().strip()
    if not raw:
        continue
    tokens = raw.split()
    command, *args = tokens

    match command:
        case "/start":
            if len(args) >= 1 and args[0].isdigit() and 2 <= int(args[0]) <= 4:
                num_players = int(args[0])
                agent_index = int(args[1]) if len(args) > 1 and args[1].isdigit() else 0
            used_tiles = [0] * 28
            hand_tiles = [0] * 28
            left_end = -1
            right_end = -1
            current_player = 0
            remaining = [7 if i < num_players else -1 for i in range(4)]
            remaining[agent_index] = 0
            state.used_tiles = used_tiles
            state.hand_tiles = hand_tiles
            state.left_end = left_end
            state.right_end = right_end
            state.remaining_counts = remaining
            print(f"Game started with {num_players} players. You are player {agent_index}.")

        case "/break":
            print("Game ended.")
            break

        case "/move":
            if len(args) != 3:
                print("Usage: /move a b side")
                continue
            try:
                a, b = int(args[0]), int(args[1])
                side = args[2]
                tile = (a, b) if (a, b) in ALL_DOMINOES else (b, a)
                idx = ALL_DOMINOES.index(tile)

                if current_player == agent_index and not hand_tiles[idx]:
                    print("Tile not in your hand.")
                    continue

                if not state.is_board_empty:
                    if side == 'l':
                        if tile[0] != state.left_end and tile[1] != state.left_end:
                            print(f"Illegal move: {tile} doesn't match left end {state.left_end}")
                            continue
                    elif side == 'r':
                        if tile[0] != state.right_end and tile[1] != state.right_end:
                            print(f"Illegal move: {tile} doesn't match right end {state.right_end}")
                            continue
                    else:
                        print("Side must be 'l' or 'r'")
                        continue

                if current_player == agent_index:
                    hand_tiles[idx] = 0
                used_tiles[idx] = 1

                if state.is_board_empty:
                    left_end, right_end = tile
                else:
                    if side == 'l':
                        left_end = tile[0] if tile[1] == state.left_end else tile[1]
                    else:
                        right_end = tile[0] if tile[1] == state.right_end else tile[1]

                state.left_end = left_end
                state.right_end = right_end
                state.remaining_counts[current_player] -= 1
                current_player = (current_player + 1) % num_players

            except Exception as e:
                print("Invalid move.", e)

        case "/pass":
            print(f"Player {current_player} passed.")
            current_player = (current_player + 1) % num_players

        case "/draw":
            if len(args) != 1 or not args[0].isdigit():
                print("Usage: /draw n")
                continue
            n = int(args[0])
            if remaining[current_player] == -1:
                print("This player is not active.")
            else:
                remaining[current_player] += n
                state.remaining_counts = remaining
                print(f"Player {current_player} drew {n} tiles. Remaining now: {remaining[current_player]}")

        case "/i_draw":
            try:
                for i in range(0, len(args), 2):
                    a, b = int(args[i]), int(args[i + 1])
                    tile = (a, b) if (a, b) in ALL_DOMINOES else (b, a)
                    idx = ALL_DOMINOES.index(tile)
                    if hand_tiles[idx]:
                        print(f"Already in hand: {tile}")
                    else:
                        hand_tiles[idx] = 1
                        remaining[agent_index] += 1
                        print(f"Added to your hand: {tile}")
            except Exception as e:
                print("Error parsing tiles:", e)
            state.hand_tiles = hand_tiles
            state.remaining_counts = remaining

        case "/agent":
            if current_player != agent_index:
                print("It's not your turn.")
            else:
                mask = state.legal_actions
                legal = [i for i, ok in enumerate(mask) if ok]
                agent.select_action(state, legal, training=False, suggestions=True)

        case "/show":
            print(state)
            print(f"Current player: {current_player}")

        case _:
            print("Unknown command.")
