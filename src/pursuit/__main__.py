from functools import reduce
from operator import mul
import numpy as np
from src.pursuit.compute_optimal_policy import choose_action, compute_policies
from src.pursuit.game import Game
from src.pursuit.reinforce import Player, bootstrap_to_optimal
from src.pursuit.state import State


def main():
    g = Game(max_stages=12, alternate=False)
    g.inc_cost = 0
    g.fail_cost = 1
    V, gamma_array, sigma_array = compute_policies(g)

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

    print("Game from cost-to-go policies:")
    g.play(gamma, sigma)
    print(f"J = {J}")


    print("Game from RL agents:")
    g = Game(max_stages=12, alternate=False)
    gamma_rl = Player(["X", "N", "E", "S", "W"], reduce(mul, g.state.board_shape), sum(g.state.lion_spaces), False)
    sigma_rl = Player(list(range(-g.lion_speed, g.lion_speed + 1)), reduce(mul, g.state.board_shape), sum(g.state.lion_spaces), True)
    bootstrap_to_optimal(g, gamma_rl, sigma_rl)
    g.play(gamma_rl, sigma_rl)
    print(f"J = {g.outcome(gamma_rl, sigma_rl)}")
    # # display the outcome from each of the Zebra's initial squares
    # V_array = np.empty((g.state.board_height, g.state.board_width))
    # for row in range(0, g.state.board_height):
    #     for col in range(0, g.state.board_width):
    #         if (col, row) in g.state.obstacles:
    #             V_array[row][col] = np.NaN
    #             continue
    #         s = State(x=col, y=row)
    #         V_array[row][col] = V[s]
    # print('Outcome of game for each starting state:')
    # print(V_array)


if __name__ == "__main__":
    main()
