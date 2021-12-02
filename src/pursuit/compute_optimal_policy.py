import itertools
import numpy as np
from scipy.linalg.basic import lstsq
from scipy.optimize import linprog

from src.pursuit.state import State

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

    result = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, options={'lstsq': True})

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


def compute_policies(game):
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
            [game.max_stages],
            range(-1, game.state.board_width + 1),
            range(-1, game.state.board_height + 1),
            range(sum(game.state.lion_spaces)),
        ):
            state = State(*state_tuple)
            V[state] = game.fail_cost

        for state_tuple in itertools.product(
            range(game.max_stages),
            [-1, game.state.board_width],
            range(game.state.board_height),
            range(sum(game.state.lion_spaces)),
        ):
            state = State(*state_tuple)
            V[state] = game.fail_cost if game.is_zebra_caught(state) else 0

        for state_tuple in itertools.product(
            range(game.max_stages),
            range(game.state.board_width),
            [-1, game.state.board_height],
            range(sum(game.state.lion_spaces)),
        ):
            state = State(*state_tuple)
            V[state] = game.fail_cost if game.is_zebra_caught(state) else 0
        


        # recursively compute cost-to-go
        for state_tuple in itertools.product(
            reversed(range(game.max_stages)),
            range(game.state.board_width),
            range(game.state.board_height),
            range(sum(game.state.lion_spaces)),
        ):
            state = State(*state_tuple)
            if (state.x, state.y) in game.state.obstacles:
                continue

            # get allowable actions from current state
            Gamma = game.zebra_action_space(state)
            Sigma = game.lion_action_space(state)

            # create matrix game
            A = np.empty((len(Gamma), len(Sigma)))
            for i, zebra_dir in enumerate(Gamma):
                for j, lion_dir in enumerate(Sigma):
                    next_state = game.next_state(zebra_dir, lion_dir, state)
                    A[i][j] = game.inc_cost + V[next_state]

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
                # compute mixed saddle point using linear programming
                V[state], y_star, z_star = solve_matrix_mixed(A)
                gamma[state] = {
                    action: y_star.item(i) for i, action in enumerate(Gamma)
                }
                sigma[state] = {
                    action: z_star.item(i) for i, action in enumerate(Sigma)
                }

        return V, gamma, sigma