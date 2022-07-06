from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Activation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module
from neuralogic.nn.module.general.rnn import RNNCell


class GRUCell(Module):
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
        arity: int = 1,
        next_name: str = "_next__positive",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_input_name = hidden_input_name

        self.arity = arity
        self.next_name = next_name

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        p_terms = [*terms, V.Z]
        h_terms = [*terms, V.T]
        input_terms = [*terms, V.T]

        r_name = f"{self.output_name}__r"
        z_name = f"{self.output_name}__z"
        n_name = f"{self.output_name}__n"
        n_helper_name = f"{self.output_name}__n_helper"
        n_helper_weighted_name = f"{self.output_name}__n_helper_w"

        h_left_name = f"{self.output_name}__left"
        h_right_name = f"{self.output_name}__right"

        next_rel = R.get(self.next_name)(V.Z, V.T)

        r = RNNCell(
            self.input_size,
            self.hidden_size,
            r_name,
            self.input_name,
            self.hidden_input_name,
            Activation.SIGMOID,
            self.arity,
            self.next_name,
        )
        z = RNNCell(
            self.input_size,
            self.hidden_size,
            z_name,
            self.input_name,
            self.hidden_input_name,
            Activation.SIGMOID,
            self.arity,
            self.next_name,
        )

        h_weight = self.hidden_size, self.hidden_size
        n_helper = R.get(n_helper_name)(h_terms) <= (
            R.get(r_name)(h_terms),
            R.get(n_helper_weighted_name)(h_terms),
        )

        n_helper_weighted = R.get(n_helper_weighted_name)(h_terms) <= (
            R.get(self.hidden_input_name)(p_terms)[h_weight],
            next_rel,
        )

        i_weight = self.hidden_size, self.input_size
        n = R.get(n_name)(h_terms) <= (R.get(self.input_name)(input_terms)[i_weight], R.get(n_helper_name)(h_terms))

        h_left = R.get(h_left_name)(h_terms) <= (-R.get(z_name)(h_terms), R.get(n_name)(h_terms))
        h_right = R.get(h_right_name)(h_terms) <= (
            R.get(z_name)(h_terms),
            R.get(self.hidden_input_name)(p_terms),
            next_rel,
        )

        h = R.get(self.output_name)(h_terms) <= (R.get(h_left_name)(h_terms), R.get(h_right_name)(h_terms))

        return [
            *r(),
            *z(),
            n_helper | Metadata(activation="elementproduct-identity"),
            n_helper.head.predicate | [Activation.IDENTITY],
            n_helper_weighted | [Activation.IDENTITY],
            n_helper_weighted.head.predicate | [Activation.IDENTITY],
            n | [Activation.TANH],
            n.head.predicate | [Activation.IDENTITY],
            h_left | Metadata(activation="elementproduct-identity"),
            h_left.head.predicate | [Activation.IDENTITY],
            h_right | Metadata(activation="elementproduct-identity"),
            h_right.head.predicate | [Activation.IDENTITY],
            h | [Activation.IDENTITY],
            h.head.predicate | [Activation.IDENTITY],
        ]


class GRU(Module):
    r"""
    One-layer Gated Recurrent Unit (GRU) module which is computed as:

    .. math::

        r_t = \sigma(\mathbf{W}_{xr} \mathbf{x}_t + \mathbf{W}_{hr} \mathbf{h}_{t-1}) \\

    .. math::

        z_t = \sigma(\mathbf{W}_{xz} \mathbf{x}_t + \mathbf{W}_{hz} \mathbf{h}_{t-1}) \\

    .. math::

        n_t = \tanh(\mathbf{W}_{xn} \mathbf{x}_t + r_t \odot (\mathbf{W}_{hn} \mathbf{h}_{t-1})) \\

    .. math::

        h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}

    where :math:`t \in (1, sequence\_length + 1)` is a time step.
    In the template, the :math:`t` is referred to as :code:`V.T`, and :math:`t - 1` is referred to as :code:`V.Z`.
    This module expresses the first equation as:

    .. code:: logtalk

        (R.<output_name>__r(<...terms>, V.T) <= (
            R.<input_name>(<...terms>, V.T)[<hidden_size>, <input_size>],
            R.<hidden_input_name>(<...terms>, V.Z)[<hidden_size>, <hidden_size>],
            R.<next_name>(V.Z, V.T),
        )) | [Activation.SIGMOID]

        R.<output_name>__r / <arity> + 1 | [Activation.IDENTITY]

    The second equation is expressed in the same way, except for a different head predicate name. The third equation is
    split into three rules. The first two computes the element-wise product -
    :math:`r_t * (\mathbf{W}_{hn} \mathbf{h}_{t-1})`.

    .. code:: logtalk

        (R.<output_name>__n_helper_weighted(<...terms>, V.T) <= (
            R.<hidden_input_name>(<...terms>, V.Z)[<hidden_size>, <hidden_size>], R.<next_name>(V.Z, V.T),
        )) | [Activation.IDENTITY],

        R.<output_name>__n_helper_weighted / (<arity> + 1) | [Activation.IDENTITY],

        (R.<output_name>__n_helper(<...terms>, V.T) <= (
            R.<output_name>__r(<..terms>, V.T), R.<>__n_helper_weighted(<...terms>, V.T)
        )) | Metadata(activation="elementproduct-identity"),

        R.<output_name>__n_helper / (<arity> + 1) | [Activation.IDENTITY],

    The third one computes the sum and applies the :math:`tanh` activation function.

    .. code:: logtalk

        (R.<output_name>__n(<...terms>, V.T) <= (
            R.<input_name>(<...terms>, V.T)[<hidden_size>, <input_size>],
            R.<output_name>__n_helper(<...terms>, V.T)
        )) | [Activation.TANH]
        R.<output_name>__n / (<arity> + 1) | [Activation.IDENTITY],

    The last equation is computed via three rules. The first two rules computes element-wise products. That is:

    .. code:: logtalk

        (R.<output_name>__left(<...terms>, V.T) <= (
            R.<output_name>__z(<...terms>, V.T), R.<output_name>__n(<...terms>, V.T)
        )) | Metadata(activation="elementproduct-identity")

        (R.<output_name>__right(<...terms>, V.T) <= (
            R.<output_name>__z(<...terms>, V.T), R.<hidden_input_name>(<...terms>, V.Z), R.<next_name>(V.Z, V.T),,
        )) | Metadata(activation="elementproduct-identity")

        R.<output_name>__left / <arity> + 1 | [Activation.IDENTITY]
        R.<output_name>__right / <arity> + 1 | [Activation.IDENTITY]

    The last output rule sums up the element-wise products.

    .. code:: logtalk

        (R.<output_name>(<...terms>, V.T) <= (
            R.<output_name>__left(<...terms>, V.T), R.<output_name>__right(<...terms>, V.T)
        )) | [Activation.IDENTITY]
        R.<output_name> / <arity> + 1 | [Activation.IDENTITY],

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
        arity: int = 1,
        next_name: str = "_next__positive",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_0_name = hidden_0_name

        self.arity = arity
        self.next_name = next_name

    def __call__(self):
        recursive_cell = GRUCell(
            self.input_size,
            self.hidden_size,
            self.output_name,
            self.input_name,
            self.output_name,
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
