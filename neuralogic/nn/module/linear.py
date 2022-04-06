from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class Linear(Module):
    r"""

    Parameters
    ----------

    in_channels : int
        Input feature size.
    out_channels : int
        Output feature size.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input name.
    activation : Activation
        Activation function of the output.
        Default: ``Activation.IDENTITY``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        input_name: str,
        activation: Activation = Activation.IDENTITY,
    ):
        self.output_name = output_name
        self.input_name = input_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I)[self.out_channels, self.in_channels]
        metadata = Metadata(activation=Activation.IDENTITY)

        return [
            (head <= R.get(self.input_name)(V.J)) | [Activation.IDENTITY],
            R.get(self.output_name) / 1 | Metadata(activation=self.activation),
        ]
