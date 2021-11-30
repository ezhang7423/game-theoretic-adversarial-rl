import numpy as np

from src.pursuit.game import Game, choose_action
from src.pursuit.state import State


def main():
    s = State(y=2)
    g = Game(state_init=s, max_stages=14, alternate=False)
    g.inc_cost = 0
    g.fail_cost = 1

    V, gamma_array, sigma_array = g.compute_policies()

    def gamma(state):
        try:
            policy = gamma_array[state]
        except KeyError:
            policy = "X"
        return choose_action(policy)

    def sigma(state):
        try:
            policy = sigma_array[state]
        except KeyError:
            policy = 0
        return choose_action(policy)

    J = g.outcome(gamma, sigma)

    g.play(gamma, sigma)
    print(f"J = {J}")

    # display the outcome from each of the Zebra's initial squares
    V_array = np.empty((g.board_height, g.board_width))
    for row in range(0, g.board_height):
        for col in range(0, g.board_width):
            if (col, row) in g.obstacles:
                V_array[row][col] = np.NaN
                continue
            s = State(x=col, y=row)
            V_array[row][col] = V[s]
    print(V_array)


if __name__ == "__main__":
    main()
