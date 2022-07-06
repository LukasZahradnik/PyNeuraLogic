from typing import List, Union

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Activation, Function
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
    activation : Union[Function, List[Function]]
        Activation function of all layers or list of activations for each layer.
        Default: ``Activation.RELU``
    """

    def __init__(
        self,
        units: List[int],
        output_name: str,
        input_name: str,
        activation: Union[Function, List[Function]] = Activation.RELU,
    ):
        self.output_name = output_name
        self.input_name = input_name

        self.units = units
        self.activation: Union[Function, List[Function]] = activation

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
            act_layer = out_layer / 1 | (Metadata(activation=self.activation[index]) if metadata is None else metadata)

            layers.append(act_layer)
            prev_layer = out_layer

        in_channels, out_channels = self.units[-2], self.units[-1]
        out = R.get(self.output_name)

        layers.append(out(V.I)[out_channels, in_channels] <= prev_layer(V.I))
        layers.append(out / 1 | (Metadata(activation=self.activation[-1]) if metadata is None else metadata))

        return layers
