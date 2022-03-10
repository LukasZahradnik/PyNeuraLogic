from typing import List

from neuralogic.core import Template
from neuralogic.utils.templates.modules import AbstractModule

from neuralogic.utils.templates.modules.gcn import GCNConv
from neuralogic.utils.templates.modules.gsage import SAGEConv
from neuralogic.utils.templates.modules.gin import GINConv
from neuralogic.utils.templates.modules.pooling import GlobalPooling


class TemplateList:
    """TemplateList is a collection of pre-defined modules"""

    def __init__(
        self,
        modules: List[AbstractModule],
        edge_name: str = "_edge",
        feature_name: str = "node_feature",
        output_name: str = "predict",
    ):
        self.modules = modules
        self.edge_name = edge_name
        self.feature_name = feature_name
        self.output_name = output_name

    def build(self, template: Template):
        if len(self.modules) == 0:
            return

        previous_names = []
        output_layer = len(self.modules)

        for i, component in enumerate(self.modules):
            name = component.name

            if i == output_layer - 1:
                component.name = self.output_name

            previous_names.append(component.build(template, i + 1, previous_names, self.feature_name, self.edge_name))

            component.name = name
