from typing import List, Sized, Iterable

from neuralogic.core import Template, Atom, Var, Metadata, Activation
from neuralogic.utils.templates.component import AbstractComponent

from neuralogic.utils.templates.gcn import GCNConv
from neuralogic.utils.templates.gsage import SAGEConv
from neuralogic.utils.templates.gin import GINConv
from neuralogic.utils.templates.pooling import GlobalPooling


class TemplateList:
    def __init__(self, modules: List[AbstractComponent], num_features: int):
        self.modules = modules
        self.num_features = num_features

    def build(self, template: Template):
        if len(self.modules) == 0:
            return

        with template.context():
            previous_names = []

            head_atom = Atom.get(AbstractComponent.features_name)(Var.X)
            next_dim = self.modules[0].in_channels
            feature_rule = head_atom[next_dim, self.num_features] <= Atom.feature(Var.X, Var.Y)

            template.add_rule(feature_rule | Metadata(activation=Activation.IDENTITY))
            template.add_rule(Atom.get(AbstractComponent.features_name) / 1 | Metadata(activation=Activation.IDENTITY))

            for i, component in enumerate(self.modules):
                next_dim = 1 if i == len(self.modules) - 1 else self.modules[i + 1].in_channels
                previous_names.append(component.build(template, i + 1, previous_names, next_dim))

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
