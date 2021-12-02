import numpy as np


from src.pursuit.state import State


class Game:
    def __init__(
        self,
        state_init=None,
        max_stages=20,
        k=0,
        x=1,
        y=2,
        lion=0,
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
            self.state = State(
                k,
                x,
                y,
                lion,
                board_shape,
                lion_spaces,
                {
                    (0, 0),
                    (0, board_shape[1] - 1),
                    (board_shape[0] - 1, 0),
                    (board_shape[0] - 1, board_shape[1] - 1),
                },
                alternate,
            )

        else:
            self.state = state_init

        # temporal and spatial boundaries
        self.max_stages = max_stages

        # lion behavior
        self.is_exact_catch = lion_spaces == "exact"
        if self.is_exact_catch:
            lion_spaces = (
                self.state.board_width - 2,
                self.state.board_height - 2,
                self.state.board_width - 2,
                self.state.board_height - 2,
            )

        if state_init is None:
            self.state.lion_spaces = lion_spaces

        self.lion_speed = lion_speed

        # alternate play?
        self.alternate = alternate
        if not self.alternate:
            self.max_stages = max_stages // 2

        self.inc_cost = 1
        self.fail_cost = np.inf

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
            if (state.x, state.y - 1) not in self.state.obstacles:
                allowed.append("N")
            if (state.x + 1, state.y) not in self.state.obstacles:
                allowed.append("E")
            if (state.x, state.y + 1) not in self.state.obstacles:
                allowed.append("S")
            if (state.x - 1, state.y) not in self.state.obstacles:
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
        if state.x < 0 or state.x >= self.state.board_width:  # escape to left  or right
            return True
        if state.y < 0 or state.y >= self.state.board_height:  # escape to top or bottom
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
            return self.state.get_lion_coord() == (state.x, state.y)

        cum_lion_spaces = [
            sum(self.state.lion_spaces[j] for j in range(i)) for i in range(4)
        ]

        if state.y < 0 and state.lion < cum_lion_spaces[1]:
            return True
        if state.x >= self.state.board_width and state.lion in range(
            cum_lion_spaces[1], cum_lion_spaces[2]
        ):
            return True
        if state.y >= self.state.board_height and state.lion in range(
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
        state.lion %= sum(self.state.lion_spaces)
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

    def play_single(self, zebra_dir, lion_dir):
        """Play a single turn and advance the game's state.

        Args:
            zebra_dir (str): The Zebra's action.
            lion_dir (int): The Lion's action.
        """
        self.state = self.next_state(zebra_dir, lion_dir)
        self.state.print_state()

    def play(self, gamma, sigma):
        """Play  the game until the end, printing the state after each turn.

        Args:
            gamma (function): Given the state, returns a pure or mixed policy for the Zebra.
            sigma (function): Given the state, returns a pure or mixed policy for the Lion.
        """
        self.state.print_state()
        while not self.is_game_over():
            zebra_dir = gamma(self.state)
            lion_dir = sigma(self.state)
            self.play_single(zebra_dir, lion_dir)
