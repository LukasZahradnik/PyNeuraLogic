from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Activation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module
from neuralogic.nn.module.general.rnn import RNNCell


class LSTMCell(Module):
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
    cell_state_0_name : str
        Predicate name to get initial cell state from.
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
        cell_state_0_name: str,
        arity: int = 1,
        next_name: str = "_next__positive",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_input_name = hidden_input_name
        self.cell_state_0_name = cell_state_0_name

        self.arity = arity
        self.next_name = next_name

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        z_terms = [*terms, V.Z]
        t_terms = [*terms, V.T]

        i_name = f"{self.output_name}__i"
        f_name = f"{self.output_name}__f"
        g_name = f"{self.output_name}__n"
        o_name = f"{self.output_name}__o"

        c_left_name = f"{self.output_name}__left"
        c_right_name = f"{self.output_name}__right"
        c_name = f"{self.output_name}__c"
        c_tanh_name = f"{self.output_name}__ctanh"

        next_rel = R.get(self.next_name)(V.Z, V.T)

        cell_args = [
            self.input_size,
            self.hidden_size,
            i_name,
            self.input_name,
            self.hidden_input_name,
            Activation.SIGMOID,
            self.arity,
            self.next_name,
        ]

        i = RNNCell(*cell_args)

        cell_args[2] = f_name
        f = RNNCell(*cell_args)

        cell_args[2] = o_name
        o = RNNCell(*cell_args)

        cell_args[2] = g_name
        cell_args[-3] = Activation.TANH
        g = RNNCell(*cell_args)

        c_left = R.get(c_left_name)(t_terms) <= (R.get(f_name)(t_terms), R.get(c_name)(z_terms), next_rel)
        c_right = R.get(c_right_name)(t_terms) <= (R.get(i_name)(t_terms), R.get(g_name)(t_terms))
        c = R.get(c_name)(t_terms) <= (R.get(c_left_name)(t_terms), R.get(c_right_name)(t_terms))
        c_tanh = R.get(c_tanh_name)(t_terms) <= (R.get(c_name)(t_terms))
        h = R.get(self.output_name)(t_terms) <= (R.get(o_name)(t_terms), R.get(c_tanh_name)(t_terms))

        return [
            *i(),
            *f(),
            *o(),
            *g(),
            c_left | Metadata(activation="elementproduct-identity"),
            c_right | Metadata(activation="elementproduct-identity"),
            c_left.head.predicate | [Activation.IDENTITY],
            c_right.head.predicate | [Activation.IDENTITY],
            c | [Activation.IDENTITY],
            (R.get(c_name)([*terms, 0]) <= R.get(self.cell_state_0_name)(terms)) | [Activation.IDENTITY],
            c.head.predicate | [Activation.IDENTITY],
            c_tanh | [Activation.TANH],
            c_tanh.head.predicate | [Activation.IDENTITY],
            h | Metadata(activation="elementproduct-identity"),
            h.head.predicate | [Activation.IDENTITY],
        ]


class LSTM(Module):
    r"""
    One-layer Long Short-Term Memory (LSTM) RNN module which is computed as:

    .. math::

        i_t = \sigma(\mathbf{W}_{xi} \mathbf{x}_t + \mathbf{W}_{hi} \mathbf{h}_{t-1})

    .. math::

        f_t = \sigma(\mathbf{W}_{xf} \mathbf{x}_t + \mathbf{W}_{hf} \mathbf{h}_{t-1})

    .. math::

        o_t = \sigma(\mathbf{W}_{xo} \mathbf{x}_t + \mathbf{W}_{ho} \mathbf{h}_{t-1})


    .. math::

        g_t = \tanh(\mathbf{W}_{xg} \mathbf{x}_t + \mathbf{W}_{hg} \mathbf{h}_{t-1}) \\

    .. math::

        c_t = f_t \odot c_{t-1} + i_t \odot g_t

    .. math::

        h_t = o_t \odot \tanh(c_t)

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
    cell_state_0_name : str
        Predicate name to get initial cell state from.
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
        cell_state_0_name: str,
        arity: int = 1,
        next_name: str = "_next__positive",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_0_name = hidden_0_name
        self.cell_state_0_name = cell_state_0_name

        self.arity = arity
        self.next_name = next_name

    def __call__(self):
        recursive_cell = LSTMCell(
            self.input_size,
            self.hidden_size,
            self.output_name,
            self.input_name,
            self.output_name,
            self.cell_state_0_name,
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
