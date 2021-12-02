# Implements vanilla policy gradient in the DARL framework
import random 
from functools import cache
import numpy as np
from src.pursuit.game import Game
from copy import deepcopy
from src.pursuit.state import State

STOP_DELTA_OBJECTIVE = (
    10  # when the change in delta is small enough to accept the policies as optimal
)


def calc_objective(p1, p2, game, silent=True):
    return 0

@cache # ended up being two slow
def full_extend(trajectory, game: Game, stage):
    if stage >= game.max_stages:
        trajectory["reward"] += game.fail_cost
        return [trajectory]

    trajectories = []
    cur_state = trajectory["state"]
    # print(game.zebra_action_space(cur_state), game.lion_action_space(cur_state))
    for z_action in game.zebra_action_space(cur_state):
        for l_action in game.lion_action_space(cur_state):
            new_trajectory = deepcopy(trajectory)
            new_trajectory["p1_actions"].append(z_action)
            new_trajectory["p2_actions"].append(l_action)

            next_state = game.next_state(z_action, l_action, cur_state)
            new_trajectory["state"] = next_state
            # next_state.print_state()
            if game.is_game_over(next_state):
                new_trajectory["reward"] += (
                    np.sign(game.is_zebra_caught(next_state) - 0.5) * game.fail_cost
                )

                trajectories.append(trajectory)
            else:
                trajectories += extend(new_trajectory, game, stage + 1)

    return trajectories

def sampled_extend(trajectory, game: Game, stage):
    if stage >= game.max_stages:
        trajectory["reward"] += game.fail_cost
        return [trajectory]

    trajectories = []
    cur_state = trajectory["state"]
    # print(game.zebra_action_space(cur_state), game.lion_action_space(cur_state))
    for z_action in random.choices(game.zebra_action_space(cur_state), k=1):
        for l_action in game.lion_action_space(cur_state):
            new_trajectory = deepcopy(trajectory)
            new_trajectory["p1_actions"].append(z_action)
            new_trajectory["p2_actions"].append(l_action)

            next_state = game.next_state(z_action, l_action, cur_state)
            new_trajectory["state"] = next_state
            # next_state.print_state()
            if game.is_game_over(next_state):
                new_trajectory["reward"] += (
                    np.sign(game.is_zebra_caught(next_state) - 0.5) * game.fail_cost
                )

                trajectories.append(trajectory)
            else:
                trajectories += sampled_extend(new_trajectory, game, stage + 1)

    return trajectories



def generate_all_trajectories(game: Game):
    return sampled_extend(
        {
            "reward": 0,  # cumulative reward
            "state": game.state,  # final state of trajectory
            "p1_actions": [],
            "p2_actions": [],
        },
        game,
        0,
    )


class Player:
    def __init__(
        self, actions, num_states_p1, num_states_p2, maximizer: bool, learning_rate=1e-2
    ) -> None:
        """Specifies a RL agent. Agents output an action given a state, after training

        Parameters
        ----------
        actions : List
            All actions available to take, e.g []
        num_states_p1 : int
            |P1 state space|
        num_states_p2 : int
            |P2 state space|
        maximizer : bool
            Is this the maximizer?
        learning_rate : float, optional
            multiplier of gradient, by default 1e-2
        """
        self.policy_matrix = np.zeros(
            (num_states_p1, num_states_p2, len(actions))
        )  # make these probability simplexes with uniform distribuition
        self.actions = actions
        self.maximizer = maximizer
        self.learning_rate = learning_rate

    def __call__(self, state: State):
        # sample an action
        flat_state = state.flattened()
        return np.random.choice(
            self.actions, 1, p=self.policy_matrix[flat_state[0], flat_state[1], :]
        )[0]

    def calc_grad_prob(self, trajectory, opponent):  # ! TODO
        pass

    def calc_grad(self, opponent, game):
        ret = np.zeros(self.policy_matrix.shape)  # calculate gradient matrix

        for trajectory in generate_all_trajectories(game):
            sum_grad_prob = self.calc_grad_prob(trajectory)  # TODO impl
            reward = sum(trajectory["rewards"])

            ret += reward * sum_grad_prob

        return ret

    def improve(self, opponent, game):  # add estimation later
        grad_objective = self.calc_grad(opponent, game)

        print("Avg Gradient Update:", np.linalg.norm(grad_objective))
        if not self.maximizer:
            grad_objective *= -1

        self.policy_matrix += self.learning_rate * grad_objective


def bootstrap_to_optimal(p1: Player, p2: Player, game: Game):
    objective_values = []  # history
    delta_objective = np.inf
    objective_values.append(calc_objective(p1, p2, game))
    while delta_objective > STOP_DELTA_OBJECTIVE:
        print("P1 training..")
        p1.improve(p2, game)
        objective_values.append(calc_objective(p1, p2, game))
        print("New objective value:", objective_values[-1])
        p2.improve(p1, game)
        objective_values.append(calc_objective(p1, p2, game))
        print("New objective value:", objective_values[-1])
        delta_objective = abs(objective_values[-1] - objective_values[-2])
    return objective_values
