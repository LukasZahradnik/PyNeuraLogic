from typing import Union

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation
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
    activation : Activation
        Activation function.
        Default: ``Activation.TANH``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    input_time_step : bool
        Include the time/iteration step as extra (last) term in the input predicate.
        Default: ``True``
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
        activation: Activation = Activation.TANH,
        arity: int = 1,
        input_time_step: bool = True,
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
        self.input_time_step = input_time_step

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        input_terms = terms

        p_terms = [*terms, V.Z]
        h_terms = [*terms, V.Y]

        if self.input_time_step:
            input_terms = [*input_terms, V.Y]

        r_name = f"{self.output_name}__r"
        z_name = f"{self.output_name}__z"
        n_name = f"{self.output_name}__n"
        n_helper_name = f"{self.output_name}__n_helper"

        z_minus_name = f"{self.output_name}__mz"
        h_left_name = f"{self.output_name}__left"
        h_right_name = f"{self.output_name}__right"

        next_rel = R.get(self.next_name)(V.Z, V.Y)

        r = RNNCell(
            self.input_size,
            self.hidden_size,
            r_name,
            self.input_name,
            self.hidden_input_name,
            Activation.SIGMOID,
            self.arity,
            self.input_time_step,
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
            self.input_time_step,
            self.next_name,
        )

        h_weight = self.hidden_size, self.hidden_size
        n_helper = R.get(n_helper_name)(h_terms) <= (
            R.get(r_name)(h_terms),
            R.get(self.hidden_input_name)(p_terms)[h_weight],
            next_rel,
        )

        i_weight = self.hidden_size, self.input_size
        n = R.get(n_name)(terms) <= (R.get(self.input_name)(input_terms)[i_weight], R.get(n_helper_name)(h_terms))

        z_minus = R.get(z_minus_name)(h_terms) <= (R.special.true[1.0].fixed(), R.get(z_name)(h_terms)[-1].fixed())
        h_left = R.get(h_left_name)(h_terms) <= (R.get(z_minus_name)(h_terms), R.get(n_name)(h_terms))
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
            n | [Activation.IDENTITY],
            n.head.predicate | [Activation.TANH],
            z_minus | [Activation.IDENTITY],
            z_minus.head.predicate | [Activation.IDENTITY],
            h_left | Metadata(activation="elementproduct-identity"),
            h_left.head.predicate | [Activation.IDENTITY],
            h_right | Metadata(activation="elementproduct-identity"),
            h_right.head.predicate | [Activation.IDENTITY],
            h | [Activation.IDENTITY],
            h.head.predicate | [Activation.IDENTITY],
        ]


class GRU(Module):
    r"""

    Parameters
    ----------

    input_size : int
        Input feature size.
    hidden_size : int
        Output and hidden feature size.
    num_layers : int
        Number of hidden layers.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input feature predicate name to get features from.
    hidden_0_name : str
        Predicate name to get initial hidden state from.
    activation : Activation
        Activation function.
        Default: ``Activation.TANH``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    input_time_step : bool
        Include the time/iteration step as extra (last) term in the input predicate.
        Default: ``True``
    next_name : str
        Predicate name to get positive integer sequence from.
        Default: ``_next__positive``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_name: str,
        input_name: str,
        hidden_0_name: str,
        activation: Activation = Activation.TANH,
        arity: int = 1,
        input_time_step: bool = True,
        next_name: str = "_next__positive",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_0_name = hidden_0_name

        self.activation = activation
        self.arity = arity
        self.next_name = next_name
        self.input_time_step = input_time_step

    def __call__(self):
        temp_output_name = f"{self.output_name}__gru_cell"

        recursive_cell = GRUCell(
            self.input_size,
            self.hidden_size,
            temp_output_name,
            self.input_name,
            temp_output_name,
            self.activation,
            self.arity,
            self.input_time_step,
            self.next_name,
        )

        next_relation = R.get(self.next_name)
        terms = [f"X{i}" for i in range(self.arity)]

        output_relation = R.get(self.output_name)

        return [
            *[next_relation(i, i + 1) for i in range(0, self.num_layers)],
            (R.get(temp_output_name)([*terms, 0]) <= R.get(self.hidden_0_name)(terms)) | [Activation.IDENTITY],
            *recursive_cell(),
            (output_relation(terms) <= R.get(temp_output_name)([*terms, self.num_layers])) | [Activation.IDENTITY],
            output_relation / self.arity | [Activation.IDENTITY],
        ]
