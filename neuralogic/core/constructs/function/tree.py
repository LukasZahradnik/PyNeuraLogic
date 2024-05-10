from neuralogic.core.constructs.function.function import Transformation, Combination, Aggregation


class FunctionalTree():
    # Leaf is always a relation

    def __init__(self):
        self.operation = None
        self.left_value = None
        self.right_value = None

    def __add__(self, other):
        n = FunctionalTree()
        n.operation = Combination.SUM
        n.left_value = self
        n.right_value = other
        return n

    def __mul__(self, other):
        n = FunctionalTree()
        n.operation = Combination.PRODUCT
        n.left_value = self
        n.right_value = other
        return n


    def get_leaves(self):
        leaves = []

        if self.left_value is not None:
            if isinstance(self.left_value, FunctionalTree):
                leaves.extend(self.left_value.get_leaves())
            else:
                leaves.extend([self.left_value])
        else:
            raise Exception("Tree has left value None.")

        if self.right_value is not None:
            if isinstance(self.right_value, FunctionalTree):
                leaves.extend(self.right_value.get_leaves())
            else:
                leaves.extend([self.right_value])

        return leaves


    def create_metadata(self):
        operations = self.note_operations()
        operations.reverse()
        metadata = self.process_operations(operations)
        return metadata


    def process_operations(self, operations):
        """
        GENERAL RULES

        #1: Aggregation can't precede transformation, transformation can't precede combination.
        #2: There can't be more than one transformation and aggregation. 
        #3: All combinations must be of the same type.
        """

        no_comb = no_trans = no_agg = 0
        last_comb = None
        metadata = []

        for op in operations:
            if isinstance(op, Combination):
                if no_trans != 0 or no_agg != 0:
                    raise Exception("Combination must precede Transformation and Aggregation.")
                if last_comb == None:
                    last_comb = op
                    metadata.append(op)
                elif last_comb != op:
                    raise Exception("Combination types must be the same.")
                no_comb += 1
            elif isinstance(op, Transformation):
                if no_trans != 0 or no_agg != 0:
                    raise Exception("There can only be one transformation present and it must precede Aggregation.")
                metadata.append(op)
                no_trans += 1
            elif isinstance(op, Aggregation):
                if no_agg != 0:
                    raise Exception("There can only be one Aggregation.")
                metadata.append(op)
                no_agg += 1
            
        return metadata
    

    def note_operations(self):
        # first search for aggregation, then transformation and combination
        operations = []
        operations.append(self.operation)

        if isinstance(self.left_value, FunctionalTree):
            operations += self.left_value.note_operations()
        if isinstance(self.right_value, FunctionalTree):
            operations += self.right_value.note_operations()

        return operations
    
    def has_combination(self):
        if isinstance(self.operation, Combination):
            return True
        if isinstance(self.left_value, FunctionalTree):
            if self.left_value.has_combination():
                return True
        if isinstance(self.right_value, FunctionalTree):
            if self.right_value.has_combination():
                return True
        return False

    def __str__(self) -> str:
        if self.left_value is not None:
            if self.right_value is not None:
                return f" {self.operation}({self.left_value}, {self.right_value}) "
            else:
                return f" {self.operation}({self.left_value}) "
        else:
            raise Exception("Tree has left value None.")
        

class FunctionContainer:
    """
    ORDER RULES
    #1: Square brackets are only used for aggregation.
    #2: If operation can be either Combination or Tranformation, Transformation is chosen.
    """

    class FunctionCallSimulator:
        def __init__(self, function):
            self.function = function

        def __getitem__(self, args):
            return self.function(args, True)

        def __call__(self, arg):
            return self.function(arg)

    # TRANSFORMATIONS ------------------------------------
    # element wise
    @property
    def sigmoid(self):
        return self.FunctionCallSimulator(self._sigmoid_private)
    @property
    def tanh(self):
        return self.FunctionCallSimulator(self._tanh_private)
    @property
    def signum(self):
        return self.FunctionCallSimulator(self._signum_private)
    @property
    def relu(self):
        return self.FunctionCallSimulator(self._relu_private)
    @property
    def leaky_relu(self):
        return self.FunctionCallSimulator(self._leaky_relu_private)
    @property
    def lukasiewicz(self):
        return self.FunctionCallSimulator(self._lukasiewicz_private)
    @property
    def exp(self):
        return self.FunctionCallSimulator(self._exp_private)
    @property
    def sqrt(self):
        return self.FunctionCallSimulator(self._sqrt_private)
    @property
    def inverse(self):
        return self.FunctionCallSimulator(self._inverse_private)
    @property
    def reverse(self):
        return self.FunctionCallSimulator(self._reverse_private)
    @property
    def log(self):
        return self.FunctionCallSimulator(self._log_private)
    
    # transformation
    @property
    def identity(self):
        return self.FunctionCallSimulator(self._identity_private)
    @property
    def transp(self):
        return self.FunctionCallSimulator(self._transp_private)
    @property
    def softmax(self):
        return self.FunctionCallSimulator(self._softmax_private)
    @property
    def sparsemax(self):
        return self.FunctionCallSimulator(self._sparsemax_private)
    @property
    def norm(self):
        return self.FunctionCallSimulator(self._norm_private)
    @property
    def slice(self):
        return self.FunctionCallSimulator(self._slice_private)
    @property
    def reshape(self):
        return self.FunctionCallSimulator(self._reshape_private)
    
    # COMBINATIONS ------------------------------------
    # aggregation
    @property
    def avg(self):
        return self.FunctionCallSimulator(self._avg_private)
    @property
    def max(self):
        return self.FunctionCallSimulator(self._max_private)
    @property
    def min(self):
        return self.FunctionCallSimulator(self._min_private)
    @property
    def sum(self):
        return self.FunctionCallSimulator(self._sum_private)
    @property
    def count(self):
        return self.FunctionCallSimulator(self._count_private)
    
    # combination
    @property
    def product(self):
        return self.FunctionCallSimulator(self._product_private)
    @property
    def elproduct(self):
        return self.FunctionCallSimulator(self._elproduct_private)
    @property
    def crosssum(self):
        return self.FunctionCallSimulator(self._crosssum_private)
    @property
    def concat(self):
        return self.FunctionCallSimulator(self._concat_private)
    @property
    def cossim(self):
        return self.FunctionCallSimulator(self._cossim_private)


    def _get_ft(self, val, op):
        n = FunctionalTree()
        n.operation = op
        n.left_value = val
        return n
    
    def _tree_has_comb(self, value):
        if isinstance(value, FunctionalTree):
            return value.has_combination()
        else:
            return False

    # TRANSFORMATIONS ------------------------------------
    # element wise
    def _sigmoid_private(self, value, agg=False):
        return self._get_ft(value, Transformation.SIGMOID)
    def _tanh_private(self, value, agg=False):
        return self._get_ft(value, Transformation.TANH)
    def _signum_private(self, value, agg=False):
        return self._get_ft(value, Transformation.SIGNUM)
    def _relu_private(self, value, agg=False):
        return self._get_ft(value, Transformation.RELU)
    def _leaky_relu_private(self, value, agg=False):
        return self._get_ft(value, Transformation.LEAKY_RELU)
    def _lukasiewicz_private(self, value, agg=False):
        return self._get_ft(value, Transformation.LUKASIEWICZ)
    def _exp_private(self, value, agg=False):
        return self._get_ft(value, Transformation.EXP)
    def _sqrt_private(self, value, agg=False):
        return self._get_ft(value, Transformation.SQRT)
    def _inverse_private(self, value, agg=False):
        return self._get_ft(value, Transformation.INVERSE)
    def _reverse_private(self, value, agg=False):
        return self._get_ft(value, Transformation.REVERSE)
    def _log_private(self, value, agg=False):
        return self._get_ft(value, Transformation.LOG)

    # transformation
    def _identity_private(self, value, agg=False):
        return self._get_ft(value, Transformation.IDENTITY)
    def _transp_private(self, value, agg=False):
        return self._get_ft(value, Transformation.TRANSP)
    def _softmax_private(self, value, agg=False):
        # comb, trans, agg
        if agg:
            op = Aggregation.SOFTMAX
        elif self._tree_has_comb(value):
            op = Transformation.SOFTMAX
        else:
            op = Combination.SOFTMAX
        return self._get_ft(value, op)
    def _sparsemax_private(self, value, agg=False):
        # comb, trans
        if self._tree_has_comb(value):
            op = Transformation.SOFTMAX
        else:
            op = Combination.SOFTMAX
        return self._get_ft(value, op)
    def _norm_private(self, value, agg=False):
        return self._get_ft(value, Transformation.NORM)
    def _slice_private(self, value, agg=False):
        return self._get_ft(value, Transformation.SLICE)
    def _reshape_private(self, value, agg=False):
        return self._get_ft(value, Transformation.RESHAPE)


    # COMBINATIONS ------------------------------------
    # aggregation
    def _avg_private(self, value, agg=False):
        # comb, agg
        return self._get_ft(value, Aggregation.AVG if agg else Combination.AVG)
    def _max_private(self, value, agg=False):
        # comb, agg
        return self._get_ft(value, Aggregation.MAX if agg else Combination.MAX)
    def _min_private(self, value, agg=False):
        # comb, agg
        return self._get_ft(value, Aggregation.MIN if agg else Combination.MIN)
    def _sum_private(self, value, agg=False):
        # comb, agg
        return self._get_ft(value, Aggregation.SUM if agg else Combination.SUM)
    def _count_private(self, value, agg=False):
        # comb, agg
        return self._get_ft(value, Aggregation.COUNT if agg else Combination.COUNT)

    # combination
    def _product_private(self, value, agg=False):
        return self._get_ft(value, Combination.PRODUCT)
    def _elproduct_private(self, value, agg=False):
        return self._get_ft(value, Combination.ELPRODUCT)
    def _sparsemax_private(self, value, agg=False):
        return self._get_ft(value, Combination.SPARSEMAX)
    def _crosssum_private(self, value, agg=False):
        return self._get_ft(value, Combination.CROSSSUM)
    def _concat_private(self, value, agg=False):
        # comb, agg
        return self._get_ft(value, Aggregation.CONCAT if agg else Combination.CONCAT)
    def _cossim_private(self, value, agg=False):
        return self._get_ft(value, Combination.COSSIM)