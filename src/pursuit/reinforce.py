# Implements vanilla policy gradient in the DARL framework
import numpy as np


class Player:
    def __init__(self, actions, num_states_p1, num_states_p2, maximizer: bool, learning_rate=1e-2) -> None:
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

    def get_action(self, state):

        # sample an action
        return np.random.choice(self.actions, 1, p=self.policy_matrix[state_idx, :])[0]

    def train(self):  # add estimation later
        grad_objective = np.zeros(self.policy_matrix.shape)  # calculate gradient matrix

        self.policy_matrix += self.learning_rate * grad_objective
