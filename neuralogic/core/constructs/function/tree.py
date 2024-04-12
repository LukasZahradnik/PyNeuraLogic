import torch
from neuralogic.core.constructs.function.function import Transformation, Combination, Aggregation


class FunctionalTree:
    # operations known: + - * ^
    # tree might have both left and right value, or none, or only left value - when function from F was called
    # this class is parent to BaseRelation! That's not ok, because each relation now has tree values, it's temporary solution.

    def __init__(self) -> None:
        self.operation = None
        self.left_value = None
        self.right_value = None
        self.combination_done = False
        self.transformation_done = False

    def __add__(self, other):
        n = FunctionalTree()
        n.operation = Combination.SUM
        n.left_value = self
        n.right_value = other
        n.combination_done = True
        return n

    def __mul__(self, other):
        n = FunctionalTree()
        n.operation = Combination.PRODUCT
        n.left_value = self
        n.right_value = other
        n.combination_done = True
        return n
    
    def copy_values_from(self, other):
        self.combination_done = other.combination_done
        self.transformation_done = other.transformation_done


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
        
    def create_metadata(self):

        operations = self.mark_operations()
        operations.reverse()    # Right eval order is comb, trans, agg
        # following line is outdated
        # metadata = self.map_operations_to_classes(operations)

        return operations


    def mark_operations(self):
        # first search for aggregation, then transformation and combination
        operations = []

        if (self.right_value == None) and (self.left_value == None):
            # leaf
            return operations
        
        elif self.right_value == None:
            # function
            operations.append(self.operation)
            return operations + self.left_value.mark_operations()
        
        else:
            # operator
            operations.append(self.operation)
            return operations + self.left_value.mark_operations() + self.right_value.mark_operations()
        

    def map_operations_to_classes(self, operations):
        # outdated function, keeping just for overview
        operation_mapping = {
            # Aggregations
            "avg_a" : Aggregation.AVG,
            "max_a" : Aggregation.MAX,
            "min_a" : Aggregation.MIN,
            "sum_a" : Aggregation.SUM,
            "count_a" : Aggregation.COUNT,
            "concat_a" : Aggregation.CONCAT,
            "softmax_a" : Aggregation.SOFTMAX,

            # Transformations element wise
            "sigmoid": Transformation.SIGMOID,
            "tanh": Transformation.TANH,
            "signum": Transformation.SIGNUM,
            "relu": Transformation.RELU,
            "leaky_relu": Transformation.LEAKY_RELU,
            "lukasiewicz": Transformation.LUKASIEWICZ,
            "exp": Transformation.EXP,
            "sqrt": Transformation.SQRT,
            "inverse": Transformation.INVERSE,
            "reverse": Transformation.REVERSE,
            "log": Transformation.LOG,

            # Transformations rest
            "identity": Transformation.IDENTITY,
            "transp": Transformation.TRANSP,
            "softmax_t": Transformation.SOFTMAX,
            "sparsemax_t": Transformation.SPARSEMAX,
            "norm": Transformation.NORM,
            "slice": Transformation.SLICE,
            "reshape": Transformation.RESHAPE,

            # Combinations - Aggregations
            "avg": Combination.AVG,
            "max": Combination.MAX,
            "min": Combination.MIN,
            "+": Combination.SUM,
            "count": Combination.COUNT,

            # Combinations rest
            "*": Combination.PRODUCT,
            "elproduct": Combination.ELPRODUCT,
            "softmax": Combination.SOFTMAX,
            "sparsemax": Combination.SPARSEMAX,
            "crosssum": Combination.CROSSSUM,
            "concat": Combination.CONCAT,
            "cossim": Combination.COSSIM,
        }

        mapped_operations = []
        for op in operations:
            op_key = op.lower()
            if op_key in operation_mapping:
                mapped_operations.append(operation_mapping[op_key])
            else:
                print(f"Warning: '{op}' is not a recognized operation and will be skipped.")

        return mapped_operations
    

class FunctionContainer:
    # TODO: more effective dictionary-like approach, maybe delete whole class and use funcions directly, instead of F.relu just relu
    # TODO: change the operation assignation order (avg_a)

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
    
    @property
    def softmax(self):
        return self.FunctionCallSimulator(self._softmax_private)
        
        
    def _avg_private(self, value):      
        if torch.is_tensor(value):
            # this is probably wrong
            return torch.mean(value)
        else:
            n = FunctionalTree()
            n.copy_values_from(value)
            if value.combination_done:
                n.operation = Aggregation.AVG
                n.combination_done = True
            else:
                n.operation = Combination.AVG
                n.combination_done = True
            n.left_value = value
            return n
        
    def _softmax_private(self, value):      
        if torch.is_tensor(value):
            # this is probably wrong
            return torch.softmax(value)
        else:
            n = FunctionalTree()
            n.copy_values_from(value)
            if value.transformation_done:
                n.operation = Aggregation.SOFTMAX
            else:
                n.operation = Transformation.SOFTMAX
                n.transformation_done = True
            n.left_value = value
            return n

    def _relu_private(self, value):
        if torch.is_tensor(value):
            return torch.relu(value)
        else:
            n = FunctionalTree()
            n.copy_values_from(value)
            n.transformation_done = True
            n.operation = Transformation.RELU
            n.left_value = value
            return n
    
    def _identity_private(self, value):
        if torch.is_tensor(value):
            return value
        else:
            n = FunctionalTree()
            n.copy_values_from(value)
            n.transformation_done = True
            n.operation = Transformation.IDENTITY
            n.left_value = value
            return n
        

         