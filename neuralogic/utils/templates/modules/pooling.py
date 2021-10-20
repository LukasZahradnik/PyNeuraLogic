from typing import List

from neuralogic.core import Relation, Template, Var, Activation, Aggregation, Metadata
from neuralogic.utils.templates.modules import AbstractModule


class GlobalPooling(AbstractModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        jumping_knowledge=(-1,),
        name=None,
        activation: Activation = Activation.SIGMOID,
        aggregation: Aggregation = Aggregation.AVG,
        parametrized: bool = False,
    ):
        super().__init__(
            name=name,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            aggregation=aggregation,
        )
        self.parametrized = parametrized
        self.jumping_knowledge = jumping_knowledge

    def build(
        self, template: Template, layer_count: int, previous_names: List[str], feature_name: str, edge_name: str
    ) -> str:
        name = f"l{layer_count}_pooling" if self.name is None else self.name

        if len(previous_names) == 0:
            previous_names = [feature_name]

        if self.parametrized:
            head_atom = Relation.get(name)(Var.X)[self.out_channels, self.in_channels]
        else:
            head_atom = Relation.get(name)[self.out_channels, self.in_channels]

        for layer in self.jumping_knowledge:
            rule = head_atom <= Relation.get(previous_names[layer])(Var.X)
            template.add_rule(rule | Metadata(aggregation=self.aggregation, activation=Activation.IDENTITY))

        template.add_rule(head_atom.predicate | Metadata(activation=self.activation))
        return name
