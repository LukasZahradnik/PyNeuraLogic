from typing import List, Iterable

import numpy as np

from neuralogic.core import Template, Relation
from neuralogic.utils.templates.modules import AbstractModule

from neuralogic.utils.templates.modules.gcn import GCNConv
from neuralogic.utils.templates.modules.gsage import SAGEConv
from neuralogic.utils.templates.modules.gin import GINConv
from neuralogic.utils.templates.modules.pooling import GlobalPooling
from neuralogic.utils.templates.modules.embedding import Embedding


class TemplateList:
    """TemplateList is a collection of pre-defined modules"""

    edge_name = "_edge"
    feature_name = "node_feature"
    output_name = "predict"

    def __init__(self, modules: List[AbstractModule]):
        self.modules = modules

    def build(self, template: Template):
        if len(self.modules) == 0:
            return

        previous_names = []

        for i, component in enumerate(self.modules):
            previous_names.append(component.build(template, i + 1, previous_names, self.feature_name, self.edge_name))
