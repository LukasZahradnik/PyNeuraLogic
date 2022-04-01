import json
from typing import Any, Dict, Optional

import numpy as np

from neuralogic.utils.visualize import draw_sample


class RawSample:
    __slots__ = "java_sample"

    def __init__(self, sample):
        self.java_sample = sample

    def draw(
        self,
        filename: Optional[str] = None,
        draw_ipython=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return draw_sample(self, filename, draw_ipython, img_type, value_detail, graphviz_path, *args, **kwargs)


class Sample(RawSample):
    __slots__ = ("id", "target", "neurons", "output_neuron", "java_sample")

    def __init__(self, sample, java_sample):
        super().__init__(sample)
        serialized_sample = json.loads(str(sample.exportToJson()))

        self.id = serialized_sample["id"]
        self.target = json.loads(serialized_sample["target"])
        self.neurons = Sample.deserialize_network(serialized_sample["network"])
        self.output_neuron = self.neurons[-1].index
        self.java_sample = java_sample

    @staticmethod
    def deserialize_network(network):
        neurons = []

        for i, neuron in enumerate(network):
            neuron_object = Neuron(neuron, i)
            neurons.append(neuron_object)

        return neurons


class Neuron:
    def __init__(self, neuron: Dict[str, Any], index):
        self.index = index
        self.name = neuron["name"]
        self.weighted = neuron["weighted"]
        self.activation = neuron.get("activation", None)
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
        name = name.split(" ")

        if len(name) == 3:
            return name[2]
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

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]
