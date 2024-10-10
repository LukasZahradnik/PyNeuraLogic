import json
from typing import Any, Dict, Optional, List

import jpype
import numpy as np

from neuralogic.core.settings.settings_proxy import SettingsProxy
from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.utils.visualize import draw_sample, draw_grounding


class Node:
    __slots__ = "substitutions", "_atom"

    def __init__(self, atom, substitutions: Dict):
        self.substitutions = substitutions
        self._atom = atom

    @property
    def value(self):
        return ValueFactory.from_java(self._atom.getRawState().getValue(), SettingsProxy.number_format())

    @property
    def gradient(self):
        return ValueFactory.from_java(self._atom.getRawState().getGradient(), SettingsProxy.number_format())

    def node_type(self):
        return self._atom.getClass().getSimpleName()

    def __str__(self):
        return str(self._atom)


class NeuralSample:
    __slots__ = "java_sample", "grounding", "_nodes"

    def __init__(self, sample, grounding):
        self.java_sample = sample
        self.grounding = grounding
        self._nodes = None

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._get_nodes()
        return self._nodes

    @property
    def target(self):
        return ValueFactory.from_java(self.java_sample.target, SettingsProxy.number_format())

    def get_nodes(self, literal, node_type=None):
        literal_name = literal.predicate.name
        literal_arity = literal.predicate.arity

        nodes = []
        nodes_by_name_vals = []

        if node_type is None:
            nodes_by_name_vals = self.nodes.values()
        elif node_type in self.nodes:
            nodes_by_name_vals = [self.nodes[node_type]]

        for nodes_by_name in nodes_by_name_vals:
            if literal_name not in nodes_by_name:
                continue

            for subs, value in nodes_by_name[literal_name].items():
                if len(subs) != literal_arity:
                    continue

                literal_subs = {}
                for term, sub in zip(literal.terms, subs):
                    term_str = str(term)

                    if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                        if term_str in literal_subs and sub != literal_subs[term_str]:
                            break
                        literal_subs[str(term)] = sub
                        continue

                    if str(term) != sub:
                        break
                else:
                    nodes.append(Node(value, literal_subs))
        return nodes

    def _get_nodes(self):
        nodes = {}

        for node in self.java_sample.query.evidence.allNeuronsTopologic:
            node_type = str(node.getClass().getSimpleName())
            name = str(node.name).strip()

            bracket = name.rfind("(")
            space = name.rfind(" ", 0, bracket if bracket != -1 else None)

            substitutions = tuple()
            if bracket != -1:
                subs = name[bracket + 1 :]
                name = name[space + 1 : bracket]

                r_bracket = subs.find(")")
                substitutions = tuple(subs[:r_bracket].split(", "))
            elif space != -1:
                name = name[space + 1 :]

            if node_type not in nodes:
                nodes[node_type] = {}

            if name not in nodes[node_type]:
                nodes[node_type][name] = {}

            nodes[node_type][name][substitutions] = node
        return nodes

    def get_fact(self, fact):
        for term in fact.terms:
            term_str = str(term)

            if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                raise ValueError(f"{fact} is not a fact")

        return self.get_nodes(fact, "FactNeuron")

    def set_fact_value(self, fact, value) -> int:
        for term in fact.terms:
            term_str = str(term)

            if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                raise ValueError(f"{fact} is not a fact")

        node = self.get_nodes(fact, "FactNeuron")

        if len(node) == 0:
            return -1

        sample_fact = node[0]._atom
        sample_fact.getRawState().setValue(value)
        sample_fact.offset.value = value

        return sample_fact.index

    def draw(
        self,
        filename: Optional[str] = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return draw_sample(self, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __str__(self):
        return str(self.java_sample)


class Neuron:
    def __init__(self, neuron: Dict[str, Any], index):
        self.index = index
        self.name = neuron["name"]
        self.weighted = neuron["weighted"]
        self.activation = neuron.get("transformation", None)
        self.inputs = neuron["inputs"]
        self.weights = neuron.get("weights", None)
        self.offset = neuron["offset"]
        self.value = neuron.get("value", None)
        self.pooling = neuron["pooling"]
        self.hook_name = Neuron.parse_hook_name(self.name)

        if self.value:
            self.value = json.loads(self.value)

        if self.weights is not None:
            self.weights = list(self.weights)

        if self.inputs is not None:
            self.inputs = list(self.inputs)

    @staticmethod
    def parse_hook_name(name: str):
        splitted_name = name.split(" ")

        if len(splitted_name) == 3:
            return splitted_name[2]
        return None


class Weight:
    def __init__(self, weight):
        self.index: int = weight.index
        self.name = str(weight.name)
        self.dimensions = tuple(weight.value.size())
        self.value = json.loads(str(weight.value.toString()))
        self.fixed = weight.isFixed

        if not isinstance(self.value, list):
            self.value = self.value

        if not self.dimensions:
            self.dimensions = (1,)

        if self.fixed:
            self.value = np.array(self.value).reshape(self.dimensions)

    @staticmethod
    def get_unit_weight() -> "Weight":
        weight = Weight.__new__(Weight)
        weight.index = 0
        weight.name = "unit"
        weight.dimensions = (1,)
        weight.value = 1.0
        weight.fixed = True

        return weight


class BuiltDataset:
    """BuiltDataset represents an already built dataset - that is, a dataset that has been grounded and neuralized."""

    __slots__ = "samples", "batch_size"

    def __init__(self, samples: List[NeuralSample], batch_size: int):
        self.samples = samples
        self.batch_size = batch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class Grounding:
    __slots__ = ("grounding",)

    def __init__(self, grounding):
        self.grounding = grounding

    def draw(
        self,
        filename: Optional[str] = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return draw_grounding(self.grounding, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __str__(self):
        return str(self.grounding)


class GroundedDataset:
    """GroundedDataset represents grounded examples that are not neuralized yet."""

    __slots__ = "_groundings", "_groundings_list", "_builder"

    def __init__(self, groundings, builder):
        self._groundings = groundings
        self._groundings_list = None
        self._builder = builder

    def _to_list(self):
        if self._groundings_list is None:
            self._groundings = self._groundings.collect(jpype.JClass("java.util.stream.Collectors").toList())
            self._groundings_list = [Grounding(g) for g in self._groundings]

    def __getitem__(self, item) -> Grounding:
        self._to_list()
        return self._groundings_list[item]

    def __len__(self) -> int:
        self._to_list()
        return len(self._groundings_list)

    def neuralize(self, *, progress: bool = False):
        self._to_list()

        if self._groundings_list is not None:
            return self._builder.neuralize(jpype.java.util.ArrayList(self._groundings).stream(), progress, len(self))
        if progress:
            return self._builder.neuralize(self._groundings, progress, len(self))
        return self._builder.neuralize(self._groundings, progress, 0)
