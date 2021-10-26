import json
import numpy as np
from py4j.java_gateway import get_field


class Sample:
    def __init__(self, sample):
        self.id = get_field(sample, "id")
        self.target = json.loads(get_field(sample, "target"))
        self.output_neuron = get_field(sample, "neuron")
        self.neurons = self.deserialize_network(get_field(sample, "network"))
        self.output_neuron = self.neurons[-1].index

    def deserialize_network(self, network):
        neurons = []

        for i, neuron in enumerate(network):
            neuron_object = Neuron(neuron, i)
            neurons.append(neuron_object)

            if neuron_object.index == self.output_neuron:
                break
        return neurons


class Neuron:
    def __init__(self, neuron, index):
        self.index = index
        self.name = get_field(neuron, "name")
        self.weighted = get_field(neuron, "weighted")
        self.activation = get_field(neuron, "activation")
        self.inputs = get_field(neuron, "inputs")
        self.weights = get_field(neuron, "weights")
        self.offset = get_field(neuron, "offset")
        self.value = get_field(neuron, "value")
        self.pooling = get_field(neuron, "pooling")
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
        self.index: int = get_field(weight, "index")
        self.name = get_field(weight, "name")
        self.dimensions = tuple(get_field(weight, "value").size())
        self.value = json.loads(get_field(weight, "value").toString())
        self.fixed = get_field(weight, "isFixed")

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
