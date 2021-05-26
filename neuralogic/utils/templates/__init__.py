from typing import List, Iterable

from neuralogic.core import Template, Atom
from neuralogic.utils.templates.modules import AbstractModule

from neuralogic.utils.templates.modules.gcn import GCNConv
from neuralogic.utils.templates.modules.gsage import SAGEConv
from neuralogic.utils.templates.modules.gin import GINConv
from neuralogic.utils.templates.modules.pooling import GlobalPooling
from neuralogic.utils.templates.modules.embedding import Embedding


class TemplateList:
    def __init__(self, modules: List[AbstractModule]):
        self.modules = modules

    def build(self, template: Template):
        if len(self.modules) == 0:
            return

        with template.context():
            previous_names = []

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
