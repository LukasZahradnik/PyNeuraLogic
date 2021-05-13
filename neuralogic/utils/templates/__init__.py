from typing import List, Sized, Iterable

from neuralogic.core import Template, Atom, Var
from neuralogic.utils.templates.component import AbstractComponent

from neuralogic.utils.templates.gcn import GCN
from neuralogic.utils.templates.gsage import GraphSAGE
from neuralogic.utils.templates.gin import GIN


class TemplateList:
    def __init__(self, components: List[AbstractComponent], out_weight_shape=None, feature_weight_shape=None):
        self.components = components
        self.out_weight_shape = out_weight_shape
        self.feature_weight_shape = feature_weight_shape

    def build(self, template: Template):
        with template.context():
            previous_name = None

            if self.feature_weight_shape is None:
                template.add_rule(Atom.get(AbstractComponent.features_name)(Var.X) <= Atom.feature(Var.X, Var.Y))

            for i, component in enumerate(self.components):
                previous_name = component.build(template, i + 1, previous_name)

            if self.out_weight_shape is not None:
                template.add_rule(Atom.predict[self.out_weight_shape] <= Atom.get(previous_name)(Var.X))
            else:
                template.add_rule(Atom.predict <= Atom.get(previous_name)(Var.X))

    def to_inputs(self, template: Template, x: Iterable, edge_index: Iterable, y: float):
        with template.context():
            query = Atom.predict[y]

            example = [Atom.edge(u, v) for u, v in edge_index]
            for i, features in enumerate(x):
                if isinstance(features, float):
                    example.append(Atom.feature(i, 0)[features])
                    continue

                for j, feature in enumerate(features):
                    example.append(Atom.feature(i, j)[feature])
        return query, example
