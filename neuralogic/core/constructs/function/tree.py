class FunctionalTree:
    # operations known: + - * ^
    # tree might have both left and right value, or none, or only left value - when function from F was called

    def __init__(self) -> None:
        self.operation = None
        self.left_value = None
        self.right_value = None

    def __add__(self, other):
        print("2: __add__ called")
        n = FunctionalTree()
        n.operation = "+"
        n.left_value = self
        n.right_value = other
        return n

    def __sub__(self, other):
        n = FunctionalTree()
        n.operation = "-"
        n.left_value = self
        n.right_value = other
        return n

    def __mul__(self, other):
        n = FunctionalTree()
        n.operation = "*"
        n.left_value = self
        n.right_value = other
        return n

    def __xor__(self, other):
        n = FunctionalTree()
        n.operation = "^"
        n.left_value = self
        n.right_value = other
        return n

    def print_tree(self) -> str:
        if self.left_value == None:
            if self.right_value == None:
                # leaf
                return self.__str__()
            else:
                print("ERROR 1!")
        elif self.right_value == None:
            # function
            return self.operation + "{" + self.left_value.print_tree() + "}"
        else:
            # operator
            return "(" + self.left_value.print_tree() + self.operation + self.right_value.print_tree() + ")"