from random import choice, randint

from src.action import TOTAL_ACTIONS
from src.agent import DominoAgent
from src.environment import DominoEnvironment, DominoAction
from src.state import ALL_DOMINOES


def main():
    """ Checking of environment logic correctness"""
    n = randint(2, 4)
    env = DominoEnvironment(num_players=n, agent_indices=[randint(0, n - 1)])
    agent = DominoAgent()
    agent.load("domino_agent.pt")
    _ = env.reset()

    while True:
        print("\n============================")
        while True:
            state = env.current_state
            mask = state.legal_actions

            hand_tiles = [i for i, held in enumerate(state.hand_tiles) if held]
            available_player_actions = [x for i in hand_tiles for x in (2 * i, 2 * i + 1)] + [TOTAL_ACTIONS - 1] # actions that can be played accordingly to hand
            available_board_actions = [i for i, avail in enumerate(mask) if avail] # actions that can be played accordingly to board

            legal_actions = sorted(set(available_player_actions) & set(available_board_actions))
            if legal_actions == [TOTAL_ACTIONS - 1] and env.draw_pile:
                _, tile = env.draw_tile()
                print(f"{ALL_DOMINOES[tile]} was drawn")
            else:
                break
        print(env)
        print(env.current_state)
        agent.select_action(env.current_state, legal_actions, training=False, suggestions=True)

        if env.current_player in env.agent_indices:
            while True:
                try:
                    action_index = int(input("Your action: "))
                    if action_index in legal_actions:
                        break
                    print("Invalid action. Try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            action_index = choice(legal_actions)
            print(f"P{env.current_player} plays {action_index}")

        next_states, rewards, done = env.step(DominoAction(action_index))

        if done:
            print("\n=== GAME OVER ===")
            print(env)
            print(f"Final rewards: {env.final_rewards}")
            break


if __name__ == "__main__":
    main()
