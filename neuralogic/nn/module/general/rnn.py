from typing import Union

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class RNNCell(Module):
    r"""

    Parameters
    ----------
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
        step: Union[str, int] = V.Y,
        next_name: str = "_rnn_next",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_input_name = hidden_input_name

        self.activation = activation
        self.arity = arity
        self.step = step
        self.next_name = next_name

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        output = R.get(self.output_name)

        rnn_rule = output([*terms, self.step]) <= (
            R.get(self.input_name)([*terms, self.step])[self.hidden_size, self.input_size],
            R.get(self.hidden_input_name)(terms)[self.hidden_size, self.hidden_size],
        )

        if self.step != 1:
            rnn_rule.body[-1].terms.append(V.Z)
            rnn_rule.body.append(R.get(self.next_name)(V.Z, self.step))

        return [
            rnn_rule | [Activation.IDENTITY],
            output / (self.arity + 1) | Metadata(activation=self.activation),
        ]


class RNN(Module):
    r""" """

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
        next_name: str = "_rnn__next",
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

    def __call__(self):
        recursive_cell = RNNCell(
            self.input_size,
            self.hidden_size,
            self.output_name,
            self.input_name,
            self.output_name,
            self.activation,
            self.arity,
            next_name=self.next_name,
        )
        input_cell = RNNCell(
            self.input_size,
            self.hidden_size,
            self.output_name,
            self.input_name,
            self.hidden_0_name,
            self.activation,
            self.arity,
            step=1,
        )

        rec_rules = recursive_cell()
        input_rules = input_cell()

        next_relation = R.get(self.next_name)
        terms = [f"X{i}" for i in range(self.arity)]

        output_relation = R.get(self.output_name)

        return [
            *[next_relation(i, i + 1) for i in range(1, self.num_layers)],
            rec_rules[0],
            input_rules[0],
            input_rules[1],
            (output_relation(terms) <= output_relation([*terms, self.num_layers])) | [Activation.IDENTITY],
            output_relation / self.arity | [Activation.IDENTITY],
        ]
