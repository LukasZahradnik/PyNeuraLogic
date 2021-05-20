from typing import List, Sized, Iterable

from neuralogic.core import Template, Atom, Var
from neuralogic.utils.templates.component import AbstractComponent

from neuralogic.utils.templates.gcn import GCNConv
from neuralogic.utils.templates.gsage import SAGEConv
from neuralogic.utils.templates.gin import GINConv
from neuralogic.utils.templates.pooling import Pooling


class TemplateList:
    def __init__(self, modules: List[AbstractComponent], num_features=None, feature_weight_shape=None):
        self.modules = modules
        self.num_features = num_features
        self.feature_weight_shape = feature_weight_shape

    def build(self, template: Template):
        with template.context():
            previous_names = []

            if self.feature_weight_shape is None:
                template.add_rule(Atom.get(AbstractComponent.features_name)(Var.X) <= Atom.feature(Var.X, Var.Y))

            for i, component in enumerate(self.modules):
                previous_names.append(component.build(template, i + 1, previous_names))

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
