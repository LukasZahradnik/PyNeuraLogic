from typing import List

from neuralogic.core import Relation, Template, Var, Metadata, Activation, Aggregation
from neuralogic.utils.templates.modules import AbstractModule


class GCNConv(AbstractModule):
    def build(
        self, template: Template, layer_count: int, previous_names: List[str], feature_name: str, edge_name: str
    ) -> str:
        name = f"l{layer_count}_gcn" if self.name is None else self.name
        previous_name = feature_name if len(previous_names) == 0 else previous_names[-1]

        head_atom = Relation.get(name)(Var.X)[self.out_channels, self.in_channels]
        layer = head_atom <= (Relation.get(previous_name)(Var.Y), Relation.get(edge_name)(Var.X, Var.Y))

        template.add_rule(layer | Metadata(aggregation=Aggregation.SUM, activation=Activation.IDENTITY))
        template.add_rule(Relation.get(name) / 1 | Metadata(activation=self.activation))

        return name
