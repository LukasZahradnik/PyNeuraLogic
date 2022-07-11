from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Activation, Function
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class RNNCell(Module):
    r"""

    Parameters
    ----------

    input_size : int
        Input feature size.
    hidden_size : int
        Output and hidden feature size.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input feature predicate name to get features from.
    hidden_input_name : str
        Predicate name to get hidden state from.
    activation : Function
        Activation function.
        Default: ``Activation.TANH``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    next_name : str
        Predicate name to get positive integer sequence from.
        Default: ``_next__positive``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_name: str,
        input_name: str,
        hidden_input_name: str,
        activation: Function = Activation.TANH,
        arity: int = 1,
        next_name: str = "_next__positive",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_input_name = hidden_input_name

        self.activation = activation
        self.arity = arity
        self.next_name = next_name

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        output = R.get(self.output_name)
        input_terms = [*terms, V.T]

        rnn_rule = output([*terms, V.T]) <= (
            R.get(self.input_name)(input_terms)[self.hidden_size, self.input_size],
            R.get(self.hidden_input_name)([*terms, V.Z])[self.hidden_size, self.hidden_size],
            R.get(self.next_name)(V.Z, V.T),
        )

        return [
            rnn_rule | Metadata(activation=self.activation),
            output / (self.arity + 1) | [Activation.IDENTITY],
        ]


class RNN(Module):
    r"""
    One-layer Recurrent Neural Network (RNN) module which is computed as:

    .. math::

        h_t = act(\mathbf{W}_{ih} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1})

    where :math:`t \in (1, sequence\_length + 1)` is a time step.
    In the template, the :math:`t` is referred to as :code:`V.T`, and :math:`t - 1` is referred to as :code:`V.Z`.
    This module expresses the first equation as:

    .. code:: logtalk

        (R.<output_name>(<...terms>, V.T) <= (
            R.<input_name>(<...terms>, V.T)[<hidden_size>, <input_size>],
            R.<hidden_input_name>(<...terms>, V.Z)[<hidden_size>, <hidden_size>],
            R.<next_name>(V.Z, V.T),
        )) | [<activation>]

        R.<output_name> / <arity> + 1 | [Activation.IDENTITY]

    Additionally, we define rules for the recursion purpose
    (the positive integer sequence :code:`R.<next_name>(V.Z, V.T)`) and the "stop condition", that is:

    .. code:: logtalk

        (R.<output_name>(<...terms>, 0) <= R.<hidden_0_name>(<...terms>)) | [Activation.IDENTITY]


    Parameters
    ----------

    input_size : int
        Input feature size.
    hidden_size : int
        Output and hidden feature size.
    sequence_length : int
        Sequence length.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input feature predicate name to get features from.
    hidden_0_name : str
        Predicate name to get initial hidden state from.
    activation : Function
        Activation function.
        Default: ``Activation.TANH``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    next_name : str
        Predicate name to get positive integer sequence from.
        Default: ``_next__positive``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sequence_length: int,
        output_name: str,
        input_name: str,
        hidden_0_name: str,
        activation: Function = Activation.TANH,
        arity: int = 1,
        next_name: str = "_next__positive",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_0_name = hidden_0_name

        self.activation = activation
        self.arity = arity
        self.next_name = next_name

    def __call__(self):
        recursive_cell = RNNCell(
            self.input_size,
            self.hidden_size,
            self.output_name,
            self.input_name,
            self.output_name,
            self.activation,
            self.arity,
            self.next_name,
        )

        next_relation = R.get(self.next_name)
        terms = [f"X{i}" for i in range(self.arity)]

        return [
            *[next_relation(i, i + 1) for i in range(0, self.sequence_length)],
            (R.get(self.output_name)([*terms, 0]) <= R.get(self.hidden_0_name)(terms)) | [Activation.IDENTITY],
            *recursive_cell(),
        ]
