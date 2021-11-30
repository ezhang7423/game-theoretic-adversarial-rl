class State:
    def __init__(self, k=0, x=1, y=1, lion=0):
        self.k = k
        self.x = x
        self.y = y
        self.lion = lion

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

    def copy(self):
        return State(self.k, self.x, self.y, self.lion)
