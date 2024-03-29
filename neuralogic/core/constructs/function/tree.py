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
    

class F:
    # all functions available for functional syntax type, includes combinations, aggregations and transformations
    # if input value is torch tensor, evaluate it directly
    # TODO: more effective dictionary-like approach, might delete operations like sum, that can be substituted using operators such as '+'
    
    def __init__(self):
        pass
    
    def identity(value):
        if torch.is_tensor(value):
            print("it is tensor apparently1")
            return torch.tensor(value)
        
        else:
            print("it is not tensor, it is tree1")
            n = FunctionalTree()
            n.operation = "identity"
            n.left_value = value
            return n
        
    def relu(value):
        if torch.is_tensor(value):
            print("it is tensor apparently2")
            return torch.relu(value)
        
        else:
            print("it is not tensor, it is tree2")
            n = FunctionalTree()
            n.operation = "relu"
            n.left_value = value
            return n
         