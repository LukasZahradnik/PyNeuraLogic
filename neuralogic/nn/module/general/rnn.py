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
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_name = output_name
        self.input_name = input_name
        self.hidden_input_name = hidden_input_name

        self.activation = activation
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        output = R.get(self.output_name)

        return [
            (
                output(terms)
                <= (
                    R.get(self.input_name)(terms)[self.hidden_size, self.input_size],
                    R.get(self.hidden_input_name)(terms)[self.hidden_size, self.hidden_size],
                )
            )
            | [Activation.IDENTITY],
            output / (self.arity) | Metadata(activation=self.activation),
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
            self.input_name,
            self.activation,
            self.arity,
        )
        input_cell = RNNCell(
            self.input_size,
            self.hidden_size,
            self.output_name,
            self.input_name,
            self.hidden_0_name,
            self.activation,
            self.arity,
        )

        rec_rules = recursive_cell()
        input_rules = input_cell()

        rec_rules[0].head.terms.append(V.Y)
        rec_rules[0].body.append(R.get(self.next_name)(V.Z, V.Y))
        rec_rules[0].body[1].terms.append(V.Z)

        input_rules[0].head.terms.append(1)
        input_rules[1].predicate.arity += 1

        next_relation = R.get(self.next_name)

        return [
            *[next_relation(i, i + 1) for i in range(self.num_layers - 1)],
            rec_rules[0],
            input_rules[0],
            input_rules[1],
        ]
