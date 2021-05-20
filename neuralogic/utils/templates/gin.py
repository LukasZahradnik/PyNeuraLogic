from typing import List

from neuralogic.core import Atom, Template, Var, Metadata, Activation, Aggregation
from neuralogic.utils.templates.component import AbstractComponent


class GINConv(AbstractComponent):
    def __init__(
        self,
        *,
        weight_shape,
        activation: Activation = Activation.RELU,
        aggregation: Aggregation = Aggregation.SUM,
        name=None,
        has_edge_attrs=True,
    ):
        super().__init__(
            weight_shape=weight_shape,
            activation=activation,
            aggregation=aggregation,
            name=name,
            has_edge_attrs=has_edge_attrs,
        )

    def build(self, template: Template, layer_count: int, previous_names: List[str]) -> str:
        name = f"l{layer_count}_gin" if self.name is None else self.name
        embed_name = f"l{layer_count}_gin_embed"
        previous_name = AbstractComponent.features_name if len(previous_names) == 0 else previous_names[-1]

        head_atom = Atom.get(embed_name)(Var.X)

        layer = head_atom <= (Atom.get(previous_name)(Var.Y), Atom.edge(Var.X, Var.Y))
        template.add_rule(layer | Metadata(aggregation=Aggregation.SUM, activation=Activation.IDENTITY))
        template.add_rule((head_atom <= Atom.get(previous_name)(Var.X)) | Metadata(activation=Activation.IDENTITY))
        template.add_rule(Atom.get(embed_name) / 1 | Metadata(activation=Activation.IDENTITY))

        layer = Atom.get(name)(Var.X)[self.weight_shape] <= head_atom[self.weight_shape]
        template.add_rule(layer | Metadata(activation=self.activation))
        template.add_rule(Atom.get(name) / 1 | Metadata(activation=self.activation))

        return name
