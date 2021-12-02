# Implements vanilla policy gradient in the DARL framework
import random
import numpy as np
from src.pursuit.game import Game
from copy import deepcopy
from src.pursuit.state import State

STOP_DELTA_OBJECTIVE = (
    1e5  # when the change in delta is small enough to accept the policies as optimal
)
ZEBRA_ACTION_ENUMERATIONS = {"X": 0, "N": 1, "E": 2, "S": 3, "W": 4}


def prob(trajectory, p1, p2, game):
    sum_prob = 0
    for p1_act, p2_act, state in zip(
        trajectory["p1_actions"], trajectory["p2_actions"], trajectory["states"]
    ):
        state_p1, state_p2 = state.flattened()

        sum_prob += (
            p1.policy_matrix[ZEBRA_ACTION_ENUMERATIONS[p1_act], state_p1, state_p2]
            + p2.policy_matrix[p2_act, state_p1, state_p2]
        )  # should be mult but add for stability. analyze later
    return sum_prob


def calc_objective(p1, p2, game, silent=True):
    trajectories = generate_all_trajectories(game)
    objective = 0
    for t in trajectories:
        objective += t["reward"] * prob(t, p1, p2, game)
    return objective * 1e-5 # ! fix this


def sampled_extend(trajectory, game: Game, stage):
    if stage >= game.max_stages:
        trajectory["reward"] += game.fail_cost
        return [trajectory]

    trajectories = []
    cur_state = trajectory["states"][-1]
    # print(game.zebra_action_space(cur_state), game.lion_action_space(cur_state))
    for z_action in random.choices(game.zebra_action_space(cur_state), k=1):
        for l_action in game.lion_action_space(cur_state):
            new_trajectory = deepcopy(trajectory)
            new_trajectory["p1_actions"].append(z_action)
            new_trajectory["p2_actions"].append(l_action)

            next_state = game.next_state(z_action, l_action, cur_state)
            new_trajectory["states"].append(next_state)
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
            "states": [game.state],  # final state of trajectory
            "p1_actions": [],
            "p2_actions": [],
        },
        game,
        0,
    )


class Player:
    def __init__(
        self,
        actions,
        num_states_p1,
        num_states_p2,
        maximizer: bool,
        learning_rate=1e-10,
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
        self.policy_matrix = np.full(
            (len(actions), num_states_p1, num_states_p2), 1 / len(actions)
        )  # make these probability simplexes with uniform distribuition
        self.actions = actions
        self.maximizer = maximizer
        self.learning_rate = learning_rate

    def __call__(self, state: State):
        # sample an action
        flat_state = state.flattened()
        return np.random.choice(
            self.actions, 1, p=self.policy_matrix[:, flat_state[0], flat_state[1]]
        )[0]

    def calc_sum_grad_prob(self, trajectory, game: Game):
        sum_grad_prob = np.zeros(self.policy_matrix.shape)
        for p1_act, p2_act, state in zip(
            trajectory["p1_actions"], trajectory["p2_actions"], trajectory["states"]
        ):
            state_p1, state_p2 = state.flattened()
            p1_act = ZEBRA_ACTION_ENUMERATIONS[p1_act]
            p2_act += game.lion_speed
            if self.maximizer:
                sum_grad_prob[p2_act, state_p1, state_p2] += 1 / (
                    self.policy_matrix[p2_act, state_p1, state_p2]
                )
            else:
                sum_grad_prob[p1_act, state_p1, state_p2] += 1 / (
                    self.policy_matrix[p1_act, state_p1, state_p2]
                )

        return sum_grad_prob

    def calc_grad(self, game):
        ret = np.zeros(self.policy_matrix.shape)  # calculate gradient matrix

        for trajectory in generate_all_trajectories(game):
            sum_grad_prob = self.calc_sum_grad_prob(trajectory, game)
            reward = trajectory["reward"]

            ret += reward * sum_grad_prob

        return ret

    def improve(self, game):  # add estimation later
        grad_objective = self.calc_grad(game)

        print(
            "Avg Gradient Update:", np.linalg.norm(self.learning_rate * grad_objective)
        )
        if self.maximizer: # ! wth
            grad_objective *= -1

        self.policy_matrix += self.learning_rate * grad_objective

        self.policy_matrix /= np.sum(self.policy_matrix, axis=0)

        # normalize policy matrix


def bootstrap_to_optimal(game: Game, p1: Player, p2: Player):
    objective_values = []  # history
    delta_objective = np.inf
    objective_values.append(calc_objective(p1, p2, game))
    print("Initial objective value (weghted):", objective_values[-1])
    # while delta_objective > STOP_DELTA_OBJECTIVE:
    i = 0
    while i < 5:
        print("P1 training..")
        p1.improve(game)
        objective_values.append(calc_objective(p1, p2, game))
        print("New objective value (weghted):", objective_values[-1])
        print("P2 training..")
        p2.improve(game)
        objective_values.append(calc_objective(p1, p2, game))
        print("New objective value (weghted):", objective_values[-1])
        delta_objective = abs(objective_values[-1] - objective_values[-2])
        i+=1
    return objective_values
