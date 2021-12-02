class State:
    def __init__(self, k=0, x=1, y=1, lion=0, board_shape=(5, 5), lion_spaces=(2, 2, 2, 2), obstacles=set(), alternate=True):
        """Game state

        Parameters
        ----------
        k : int, optional
            Stage of game (time), by default 0
        x : int, optional
            X-coord of zebra, by default 1
        y : int, optional
            Y-coord of zebra, by default 1
        lion : int, optional
            Offset of lion, by default 0
        board_shape: tuple, optional
        lion_spaces: tuple, optional
            the amount of nodes that the lion can move between, in order from (bottom, right, top, left)
        obstacles: set, optional
            obstacles that the zebra can't go through
        """
        self.k = k
        self.x = x
        self.y = y
        self.lion = lion
        self.board_shape = board_shape
        self.board_height = board_shape[0]
        self.board_width = board_shape[1]
        self.lion_spaces = lion_spaces
        self.obstacles = obstacles
        self.alternate = alternate

    def __repr__(self):
        return str((self.k, self.x, self.y, self.lion))

    def __eq__(self, other):
        return (self.k, self.x, self.y, self.lion) == (
            other.k,
            other.x,
            other.y,
            other.lion,
        )

    def __hash__(self):
        return hash((self.k, self.x, self.y, self.lion))

    def get_lion_coord(self):
        """Compute the Lion's current position with respect to the game board coordinates.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.

        Returns:
            (int, int): The Lion's (x,y) coordinates
        """

        cum_lion_spaces = [sum(self.lion_spaces[j] for j in range(i)) for i in range(4)]

        if self.lion < cum_lion_spaces[1]:
            offset = self.lion
            return (1 + offset, -1)
        elif self.lion in range(cum_lion_spaces[1], cum_lion_spaces[2]):
            offset = self.lion - cum_lion_spaces[1]
            return (self.board_width, 1 + offset)
        elif self.lion in range(cum_lion_spaces[2], cum_lion_spaces[3]):
            offset = self.lion - cum_lion_spaces[2]
            return (self.board_width - 2 - offset, self.board_height)
        else:
            offset = self.lion - cum_lion_spaces[3]
            return (-1, self.board_height - 2 - offset)

    def flattened(self):
        """Return a flattened form of the state, where the zebra's 2D coordinates are flattened into a vector
        returns (flat_zebra_coords, lion_offset)
        """
        return (self.x * self.board_shape[0] + self.y, self.lion)

    def print_state(self):
        """Print a graphical depiction of the state to the terminal.

        Args:
            state_in (State, optional): The current state. Defaults to the game's state.
        """

        if self.alternate:
            print(f"Stage {self.k // 2}.{self.k%2}")
        else:
            print(f"Stage {self.k}")

        board_str = ""
        for row in range(-1, self.board_height + 1):
            row_str = ""
            for col in range(-1, self.board_width + 1):
                # mark zebra and lion poistions
                is_zebra = (col, row) == (self.x, self.y)
                is_lion = (col, row) == self.get_lion_coord()
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

        print(board_str)

    def copy(self):
        return State(self.k, self.x, self.y, self.lion, self.board_shape, self.lion_spaces, self.obstacles.copy(), self.alternate)
