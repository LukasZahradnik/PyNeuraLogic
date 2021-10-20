from typing import List, Iterable

from neuralogic.core import Template, Relation
from neuralogic.utils.templates.modules import AbstractModule

from neuralogic.utils.templates.modules.gcn import GCNConv
from neuralogic.utils.templates.modules.gsage import SAGEConv
from neuralogic.utils.templates.modules.gin import GINConv
from neuralogic.utils.templates.modules.pooling import GlobalPooling
from neuralogic.utils.templates.modules.embedding import Embedding


class TemplateList:
    """TemplateList is a collection of pre-defined modules"""
    edge_name = "edge"
    feature_name = "node_feature"
    output_name = "predict"

    def __init__(self, modules: List[AbstractModule]):
        self.modules = modules

    def build(self, template: Template):
        if len(self.modules) == 0:
            return

        with template.context():
            previous_names = []

            for i, component in enumerate(self.modules):
                previous_names.append(
                    component.build(template, i + 1, previous_names, self.feature_name, self.edge_name)
                )

    @staticmethod
    def to_inputs(template: Template, x: Iterable, edge_index: Iterable, y, y_mask):
        with template.context():
            if y is None:
                queries = [Relation.get(TemplateList.output_name)]
            else:
                queries = [
                    Relation.get(TemplateList.output_name)(*term if isinstance(term, list) else term)[value]
                    for term, value in zip(y_mask, y)
                ]
            example = [
                Relation.get(TemplateList.edge_name)(int(u), int(v))[1] for u, v in zip(edge_index[0], edge_index[1])
            ]

            for i, features in enumerate(x):
                example.append(Relation.get(TemplateList.feature_name)(i)[features])
        return queries, example
