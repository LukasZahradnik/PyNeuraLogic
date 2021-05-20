from typing import List

from neuralogic.core import Atom, Template, Var, Activation, Aggregation, Metadata
from neuralogic.utils.templates.component import AbstractComponent


class Pooling(AbstractComponent):
    def __init__(
        self,
        *,
        layers=(-1,),
        name=None,
        weight_shape=None,
        activation: Activation = Activation.SIGMOID,
        aggregation: Aggregation = Aggregation.AVG,
    ):
        super().__init__(name=name, weight_shape=weight_shape, activation=activation, aggregation=aggregation)
        self.layers = layers

    def build(self, template: Template, layer_count: int, previous_names: List[str]):
        name = f"l{layer_count}_pooling" if self.name is None else self.name

        if len(previous_names) == 0:
            previous_names = [AbstractComponent.features_name]

        head_atom = Atom.get(name)

        if self.weight_shape is not None:
            head_atom = head_atom[self.weight_shape]

        for layer in self.layers:
            rule = head_atom <= Atom.get(previous_names[layer])(Var.X)
            template.add_rule(rule | Metadata(aggregation=self.aggregation, activation=Activation.IDENTITY))

        template.add_rule(Atom.get(name) / 0 | Metadata(activation=self.activation))
        return name
