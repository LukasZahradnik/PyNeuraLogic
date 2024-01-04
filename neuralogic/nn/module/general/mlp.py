from typing import List, Union, Sequence

from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation
from neuralogic.core.constructs.factories import R
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
    activation : Union[Transformation, List[Transformation]]
        Activation function of all layers or list of activations for each layer.
        Default: ``Transformation.RELU``
    arity : int
        Default: ``-1``
    """

    def __init__(
        self,
        units: List[int],
        output_name: str,
        input_name: str,
        activation: Union[Transformation, List[Transformation]] = Transformation.RELU,
        arity: int = 1,
    ):
        self.output_name = output_name
        self.input_name = input_name

        self.units = units
        self.activation: Union[Transformation, List[Transformation]] = activation
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        layers = []

        prev_layer = R.get(self.input_name)
        iters = len(self.units) - 1

        if isinstance(self.activation, Sequence):
            metadata = [Metadata(transformation=act) for act in self.activation]
            metadata.extend([Metadata(transformation=Transformation.IDENTITY)] * (iters - len(metadata)))
        else:
            metadata = [Metadata(transformation=self.activation)] * (iters + 1)

        for index in range(0, iters, 2):
            in_channels, out_channels = self.units[index], self.units[index + 1]

            head_predicate = R.get(f"{self.output_name}__mlp{index // 2}")
            body_literal = prev_layer(terms)[out_channels, in_channels]
            head_literal = head_predicate(terms)

            body_metadata = metadata[index]
            head_metadata = metadata[index + 1]

            if body_metadata is None:
                if index < len(self.activation):
                    body_metadata = Metadata(transformation=self.activation[index])
                else:
                    body_metadata = [Transformation.IDENTITY]

            if index + 2 < len(self.units):
                in_channels, out_channels = self.units[index + 1], self.units[index + 2]
                head_literal = head_literal[out_channels, in_channels]

            layers.append((head_literal <= body_literal) | body_metadata)
            layers.append(head_predicate / self.arity | head_metadata)
            prev_layer = head_predicate

        layers[-2].head.predicate.name = self.output_name
        layers[-1].predicate.name = self.output_name

        return layers
