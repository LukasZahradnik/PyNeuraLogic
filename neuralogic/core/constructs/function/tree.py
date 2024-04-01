import torch


class FunctionalTree:
    # operations known: + - * ^
    # tree might have both left and right value, or none, or only left value - when function from F was called
    # this class is parent to BaseRelation! That's not ok, because each relation now has tree values, it's temporary solution.

    def __init__(self) -> None:
        self.operation = None
        self.left_value = None
        self.right_value = None

    def __add__(self, other):
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
    

class FunctionContainer:
    # all functions available for functional syntax type, includes combinations, aggregations and transformations
    # if input value is torch tensor, evaluate it directly
    # TODO: more effective dictionary-like approach, might delete operations like sum, that can be substituted using operators such as '+'
    
    class FunctionCallSimulator:
        def __init__(self, function):
            self.function = function

        def __getitem__(self, args):
            return self.function(args)

        def __call__(self, arg):
            return self.function(arg)

    @property
    def identity(self):
        return self.FunctionCallSimulator(self._identity_private)
    
    @property
    def relu(self):
        return self.FunctionCallSimulator(self._relu_private)
    
    @property
    def avg(self):
        return self.FunctionCallSimulator(self._avg_private)
        
        
    def _avg_private(self, value):      
        if torch.is_tensor(value):
            # this is probably wrong
            return torch.mean(value)
        else:
            n = FunctionalTree()
            n.operation = "avg"
            n.left_value = value
            return n

    def _relu_private(self, value):
        if torch.is_tensor(value):
            return torch.relu(value)
        else:
            n = FunctionalTree()
            n.operation = "relu"
            n.left_value = value
            return n
    
    def _identity_private(self, value):
        if torch.is_tensor(value):
            return value
        else:
            n = FunctionalTree()
            n.operation = "identity"
            n.left_value = value
            return n
         