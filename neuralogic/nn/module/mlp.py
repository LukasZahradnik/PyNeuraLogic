from typing import List, Union

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class MLP(Module):
    r"""

    Parameters
    ----------

    units : List[int]
        List of layer sizes.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input name.
    activation : Union[Activation, List[Activation]]
        Activation function of all layers or list of activations for each layer.
        Default: ``Activation.RELU``
    """

    def __init__(
        self,
        units: List[int],
        output_name: str,
        input_name: str,
        activation: Union[Activation, List[Activation]] = Activation.RELU,
    ):
        self.output_name = output_name
        self.input_name = input_name

        self.units = units
        self.activation: Union[Activation, List[Activation]] = activation

    def __call__(self):
        layers = []

        prev_layer = R.get(self.input_name)

        if isinstance(self.activation, list):
            metadata = None
        else:
            metadata = Metadata(activation=self.activation)

        for index, (in_channels, out_channels) in enumerate(zip(self.units[:-1], self.units[1:])):
            out_layer = R.get(f"{self.output_name}__{index}")
            layers.append(out_layer(V.I)[out_channels, in_channels] <= prev_layer(V.I))

            if isinstance(self.activation, list):
                layers.append(out_layer / 1 | Metadata(activation=self.activation[index]))
            else:
                layers.append(out_layer / 1 | metadata)
            prev_layer = out_layer

        in_channels, out_channels = self.units[-2], self.units[-1]
        layers.append(R.get(self.output_name)[out_channels, in_channels] <= prev_layer(V.I))

        if isinstance(self.activation, list):
            layers.append(R.get(self.output_name) / 1 | Metadata(activation=self.activation[-1]))
        else:
            layers.append(R.get(self.output_name) / 1 | metadata)
        return layers
