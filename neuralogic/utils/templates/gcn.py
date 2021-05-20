from typing import List

from neuralogic.core import Atom, Template, Var, Metadata, Activation, Aggregation
from neuralogic.utils.templates.component import AbstractComponent


class GCNConv(AbstractComponent):
    def build(self, template: Template, layer_count: int, previous_names: List[str]) -> str:
        name = f"l{layer_count}_gcn" if self.name is None else self.name
        previous_name = AbstractComponent.features_name if len(previous_names) == 0 else previous_names[-1]

        head_atom = Atom.get(name)(Var.X)
        layer = head_atom[self.weight_shape] <= (Atom.get(previous_name)(Var.Y), Atom.edge(Var.X, Var.Y))

        template.add_rule(layer | Metadata(aggregation=Aggregation.SUM, activation=Activation.IDENTITY))
        template.add_rule(Atom.get(name) / 1 | Metadata(activation=self.activation))

        return name
