import json
from typing import Any, Dict, Optional, List

import jpype
import numpy as np

from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.utils.visualize import draw_sample, draw_grounding


class NeuralSample:
    __slots__ = "java_sample", "fact_cache", "grounding"

    def __init__(self, sample, grounding):
        self.java_sample = sample
        self.grounding = grounding
        self.fact_cache = {}

    @property
    def target(self):
        return ValueFactory.from_java(self.java_sample.target)

    def _find_fact(self, fact_str):
        for sample_fact in self.java_sample.query.evidence.allNeuronsTopologic:
            if str(sample_fact.getClass().getSimpleName()) != "FactNeuron":
                continue

            name = str(sample_fact.name).strip()
            space_index = name.rfind(" ")

            while True:
                if space_index == -1:
                    break

                if name[space_index - 1] == ",":
                    space_index = name.rfind(" ", __end=space_index)
                    continue

                if name[space_index + 1 :] == fact_str:
                    self.fact_cache[fact_str] = sample_fact
                    return sample_fact
                break

        return None

    def get_fact(self, fact):
        fact_str = fact.predicate.to_str()

        if fact_str in self.fact_cache:
            return self.fact_cache[fact_str]
        return self._find_fact(fact_str)

    def set_fact_value(self, fact, value) -> int:
        fact_str = fact.predicate.to_str()
        sample_fact = self.fact_cache[fact_str] if fact_str in self.fact_cache else self._find_fact(fact_str)
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

    def draw_grounding(
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


class GroundedDataset:
    """GroundedDataset represents grounded examples that are not neuralized yet."""

    __slots__ = "length", "_groundings", "_groundings_list", "_builder"

    def __init__(self, groundings, length, builder):
        self.length = length
        self._groundings = groundings
        self._groundings_list = None
        self._builder = builder

    def __getitem__(self, item):
        if self._groundings_list is None:
            self._groundings = self._groundings.collect(jpype.JClass("java.util.stream.Collectors").toList())
            self._groundings_list = [Grounding(g) for g in self._groundings]
        return self._groundings_list[item]

    def neuralize(self, progress: bool):
        if self._groundings_list is not None:
            return self._builder.neuralize(self._groundings.stream(), progress, self.length)
        return self._builder.neuralize(self._groundings, progress, self.length)
