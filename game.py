import itertools

import numpy as np
from scipy.optimize import linprog

from state import State


class Game:
    def __init__(
        self,
        state_init=None,
        max_stages=20,
        board_shape=(5, 5),
        lion_spaces=(2, 2, 2, 2),
        lion_speed=1,
        alternate=True,
    ):
        """Construct a new Zebra-Lion game object.

        Args:
            state_init (State, optional): The initial state. Defaults to None.
            max_stages (int, optional): The maximum length of the game. Defaults to 20.
            board_shape (tuple, optional): The board width and height. Defaults to (5, 5).
            lion_spaces (variable, optional): If this is a tuple, then it gives the number of spaces
                for the Lion along the north, east, south, and west sides respectively.
                Alternatively, it can be specified as the string "exact", in which the number of
                spaces along each side is the same as the number of board squares (minus the
                corners) and the Lion must occupy the exact square as the Zebra to catch it.
                Defaults to (2, 2, 2, 2).
            lion_speed (int, optional): The maximum number of spaces the Lion can move in either
                direction in a single stage. Defaults to 1.
            alternate (bool, optional): True if choices alternate (opposed to being simultaneous).
                Defaults to True.
        """
        if state_init is None:
            self.state = State()
        else:
            self.state = State(state_init.k, state_init.x, state_init.y, state_init.lion)

        # aliases for state components
        self.k = self.state.k
        self.x = self.state.x
        self.y = self.state.y
        self.lion = self.state.lion

        # temporal and spatial boundaries
        self.max_stages = max_stages
        self.board_width = board_shape[0]
        self.board_height = board_shape[1]

        # lion behavior
        self.is_exact_catch = lion_spaces == "exact"
        if self.is_exact_catch:
            lion_spaces = (
                self.board_width - 2,
                self.board_height - 2,
                self.board_width - 2,
                self.board_height - 2,
            )
        self.lion_spaces = lion_spaces
        self.lion_speed = lion_speed

        # obstacles that the zebra can't go through
        self.obstacles = {
            (0, 0),
            (0, self.board_height - 1),
            (self.board_width - 1, 0),
            (self.board_width - 1, self.board_height - 1),
        }

        # alternate play?
        self.alternate = alternate
        self.inc_cost = 1
        self.fail_cost = np.inf

    def print_game(self, state_in=None):
        """Print a graphical depiction of the state to the terminal.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        board_str = ""
        for row in range(-1, self.board_height + 1):
            row_str = ""
            for col in range(-1, self.board_width + 1):
                # mark zebra and lion poistions
                is_zebra = (col, row) == (state.x, state.y)
                is_lion = (col, row) == self.get_lion_coord(state)
                if is_zebra and is_lion:
                    row_str += "X"
                elif is_zebra:
                    row_str += "Z"
                elif is_lion:
                    row_str += "L"
                # mark game field and obstacles
                elif row in [-1, self.board_height] or col in [-1, self.board_width]:
                    row_str += " "
                elif (col, row) in self.obstacles:
                    row_str += "#"
                else:
                    row_str += "-"
            board_str += f"{row_str}\n"

        print(f"Stage {state.k}")
        print(board_str)

    def get_lion_coord(self, state_in=None):
        """Compute the Lion's current position with respect to the game board coordinates.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.

        Returns:
            (int, int): The Lion's (x,y) coordinates
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        cum_lion_spaces = [sum(self.lion_spaces[j] for j in range(i)) for i in range(4)]

        if state.lion < cum_lion_spaces[1]:
            offset = state.lion
            return (1 + offset, -1)
        elif state.lion in range(cum_lion_spaces[1], cum_lion_spaces[2]):
            offset = state.lion - cum_lion_spaces[1]
            return (self.board_width, 1 + offset)
        elif state.lion in range(cum_lion_spaces[2], cum_lion_spaces[3]):
            offset = state.lion - cum_lion_spaces[2]
            return (self.board_width - 2 - offset, self.board_height)
        else:
            offset = state.lion - cum_lion_spaces[3]
            return (-1, self.board_height - 2 - offset)

    def zebra_action_space(self, state_in=None):
        """Return the action space of the Zebra from the current state.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.

        Returns:
            list: The possible action.
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        allowed = ["X"]

        if not self.alternate or state.k % 2 == 0:
            if (state.x, state.y - 1) not in self.obstacles:
                allowed.append("N")
            if (state.x + 1, state.y) not in self.obstacles:
                allowed.append("E")
            if (state.x, state.y + 1) not in self.obstacles:
                allowed.append("S")
            if (state.x - 1, state.y) not in self.obstacles:
                allowed.append("W")

        return allowed

    def lion_action_space(self, state_in=None):
        """Return the action space of the Lion from the current state.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.

        Returns:
            list: The possible action.
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        allowed = [0]

        if not self.alternate or state.k % 2 == 1:
            allowed = range(-self.lion_speed, self.lion_speed + 1)

        return allowed

    def is_game_over(self, state_in=None):
        """Return whether the game is in a terminal state.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.

        Returns:
            bool: True if the game is in a terminal state.
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        if state.k > self.max_stages:  # timeout
            return True
        if state.x < 0 or state.x >= self.board_width:  # escape to left  or right
            return True
        if state.y < 0 or state.y >= self.board_height:  # escape to top or bottom
            return True
        return False

    def is_zebra_caught(self, state_in=None):  # sourcery skip: extract-method
        """Return whether the zebra is caught.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.

        Returns:
            bool: True if the Zebra is caught.
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        if self.is_exact_catch:
            return self.get_lion_coord(state) == (state.x, state.y)

        cum_lion_spaces = [sum(self.lion_spaces[j] for j in range(i)) for i in range(4)]

        if state.y < 0 and state.lion < cum_lion_spaces[1]:
            return True
        if state.x >= self.board_width and state.lion in range(
            cum_lion_spaces[1], cum_lion_spaces[2]
        ):
            return True
        if state.y >= self.board_height and state.lion in range(
            cum_lion_spaces[2], cum_lion_spaces[3]
        ):
            return True
        if state.x < 0 and state.lion >= cum_lion_spaces[3]:
            return True

        return False

    def next_state(self, zebra_dir, lion_dir, state_in=None):
        """Compute the next state when the given actions are chosen

        Args:
            zebra_dir (str): The direction that the Zebra moves.
            lion_dir (int): The number of squares the Lion moves clockwise.
            state_in (State, optional): The current state. Defaults to the game's state.

        Returns:
            State: The new state.
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        assert zebra_dir in self.zebra_action_space(state)
        assert lion_dir in self.lion_action_space(state)

        delta_z = {
            "N": (0, -1),
            "E": (1, 0),
            "S": (0, 1),
            "W": (-1, 0),
            "X": (0, 0),
        }

        state.x += delta_z[zebra_dir][0]
        state.y += delta_z[zebra_dir][1]
        state.lion += lion_dir
        state.lion %= sum(self.lion_spaces)
        state.k += 1

        return state

    def outcome(self, gamma, sigma, state_in=None):
        """Compute the outcome of the game from the given state with the given polices.

        Args:
            gamma (function): Given the state, returns a pure or mixed policy for the Zebra.
            sigma (function): Given the state, returns a pure or mixed policy for the Lion.
            state_in (State, optional): The inital state. Defaults to the game's state.

        Returns:
            float: The outcome.
        """
        if state_in is None:
            state_in = self.state
        state = state_in.copy()

        J = 0
        while not self.is_game_over(state):
            zebra_dir = gamma(state)
            lion_dir = sigma(state)
            state = self.next_state(zebra_dir, lion_dir, state)
            J += self.inc_cost

        if self.is_zebra_caught(state) or state.k >= self.max_stages:
            J = self.fail_cost

        return J

    def compute_policies(self):
        """Compute the optimal values and policies.

        The recursive algorithm for computing the cost-to-go function is used. The boundary
        conditions are defined at one stage beyond the state limit, and the Zebra taking one step
        outside of the board in any direction. At each state, the pure security polices are computed
        first. If a pure saddle point does not exist, then a mixed saddle point is found using the
        SciPy linprog function.

        Returns:
            dict: Maps states to the cost-to-go from that state.
            dict: Maps states to the optimal policy for the Zebra from that state.
            dict: Maps states to the optimal policy for the Lion from that state.
        """
        V = {}
        gamma = {}
        sigma = {}

        # boundary conditions
        for state_tuple in itertools.product(
            [self.max_stages],
            range(-1, self.board_width + 1),
            range(-1, self.board_height + 1),
            range(sum(self.lion_spaces)),
        ):
            state = State(*state_tuple)
            V[state] = self.fail_cost

        for state_tuple in itertools.product(
            range(self.max_stages),
            [-1, self.board_width],
            range(self.board_height),
            range(sum(self.lion_spaces)),
        ):
            state = State(*state_tuple)
            V[state] = self.fail_cost if self.is_zebra_caught(state) else 0

        for state_tuple in itertools.product(
            range(self.max_stages),
            range(self.board_width),
            [-1, self.board_height],
            range(sum(self.lion_spaces)),
        ):
            state = State(*state_tuple)
            V[state] = self.fail_cost if self.is_zebra_caught(state) else 0

        # recursively compute cost-to-go
        for state_tuple in itertools.product(
            reversed(range(self.max_stages)),
            range(self.board_width),
            range(self.board_height),
            range(sum(self.lion_spaces)),
        ):
            state = State(*state_tuple)
            if (state.x, state.y) in self.obstacles:
                continue

            # get allowable actions from current state
            Gamma = self.zebra_action_space(state)
            Sigma = self.lion_action_space(state)

            # create matrix game
            A = np.empty((len(Gamma), len(Sigma)))
            for i, zebra_dir in enumerate(Gamma):
                for j, lion_dir in enumerate(Sigma):
                    next_state = self.next_state(zebra_dir, lion_dir, state)
                    A[i][j] = self.inc_cost + V[next_state]

            # solve matrix game
            i_star = np.argmin(np.max(A, axis=1))
            j_star = np.argmax(np.min(A, axis=0))
            V_under = np.max(A, axis=1)[i_star]
            V_over = np.min(A, axis=0)[j_star]

            # check whether saddle point was found
            if V_under == V_over:
                gamma[state] = Gamma[i_star]
                sigma[state] = Sigma[j_star]
                V[state] = V_under
            else:
                # compute mixed sadle point using linear programming
                V[state], y_star, z_star = solve_matrix_mixed(A)
                gamma[state] = {
                    action: y_star.item(i) for i, action in enumerate(Gamma)
                }
                sigma[state] = {
                    action: z_star.item(i) for i, action in enumerate(Sigma)
                }

        return V, gamma, sigma

    def play_single(self, zebra_dir, lion_dir):
        """Play a single turn and advance the game's state.

        Args:
            zebra_dir (str): The Zebra's action.
            lion_dir (int): The Lion's action.
        """
        self.state = self.next_state(zebra_dir, lion_dir)
        self.print_game()

    def play(self, gamma, sigma):
        """Play  the game until the end, printing the state after each turn.

        Args:
            gamma (function): Given the state, returns a pure or mixed policy for the Zebra.
            sigma (function): Given the state, returns a pure or mixed policy for the Lion.
        """
        self.print_game()
        while not self.is_game_over():
            zebra_dir = gamma(self.state)
            lion_dir = sigma(self.state)
            self.play_single(zebra_dir, lion_dir)


def choose_action(policy):
    """Choose an action according to the policy.

    To implement a mixed policy, the input policy should be a dict that maps actions to its
    probability. A random action is then returned, using the corresponding probabilites as weights.

    To implement a pure policy, the input policy should be the action itself. It is then simply
    returned.

    Args:
        policy (variable type): A single action, or a dict that maps actions to probabilities.

    Returns:
        variable type: The action.
    """
    if not isinstance(policy, dict):
        return policy

    actions = list(policy.keys())
    weights = np.array([policy[a] for a in actions])

    return np.random.choice(actions, p=weights / sum(weights))


def mixed_minmax(A):
    """Compute a mixed security policy for the minimizer of a matrix game.

    Args:
        A (2-D array): The matrix game.

    Returns:
        float: Mixed security level.
        1-D array: Mixed security policy.
    """
    y_length = A.shape[0]
    z_length = A.shape[1]

    x_length = y_length + 1

    c = np.zeros(x_length)
    c[0] = 1

    A_ub = np.block([-np.ones((z_length, 1)), A.T])
    b_ub = np.zeros(z_length)

    A_eq = np.ones((1, x_length))
    A_eq[0][0] = 0
    b_eq = 1

    bounds = [(None, None)] + [(0, None)] * y_length

    result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)

    policy = result.x[1:].T
    value = result.x[0]

    return policy, value


def solve_matrix_mixed(A):
    """Compute a mixed saddle point for a matrix game.

    Args:
        A (2-D array): The matrix game.

    Returns:
        float: Mixed value for the game.
        1-D array: Mixed security policy for the minimizer.
        1-D array: Mixed security policy for the maximizer.
    """
    y_star, value = mixed_minmax(A)

    # player 2 behaves oppositely
    z_star, _ = mixed_minmax(-A.T)

    return value, y_star, z_star
