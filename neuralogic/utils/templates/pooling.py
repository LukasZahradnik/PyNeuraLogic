from typing import List

from neuralogic.core import Atom, Template, Var, Activation, Aggregation, Metadata
from neuralogic.utils.templates.component import AbstractComponent


class GlobalPooling(AbstractComponent):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        layers=(-1,),
        name=None,
        activation: Activation = Activation.SIGMOID,
        aggregation: Aggregation = Aggregation.AVG,
    ):
        super().__init__(
            name=name,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            aggregation=aggregation,
        )
        self.layers = layers

    def build(self, template: Template, layer_count: int, previous_names: List[str]) -> str:
        name = f"l{layer_count}_pooling" if self.name is None else self.name

        if len(previous_names) == 0:
            previous_names = [AbstractComponent.features_name]

        head_atom = Atom.get(name)[self.out_channels, self.in_channels]

        for layer in self.layers:
            rule = head_atom <= Atom.get(previous_names[layer])(Var.X)
            template.add_rule(rule | Metadata(aggregation=self.aggregation, activation=Activation.IDENTITY))

        template.add_rule(Atom.get(name) / 0 | Metadata(activation=self.activation))
        return name
