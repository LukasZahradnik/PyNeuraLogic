from neuralogic import get_neuralogic, get_gateway
from neuralogic.core.settings import Settings
from neuralogic.core.sources import Sources

import json
from enum import Enum
from typing import List, Optional
from py4j.java_gateway import get_field


def stream_to_list(stream) -> List:
    return list(stream.collect(get_gateway().jvm.java.util.stream.Collectors.toList()))


class Backend(Enum):
    DYNET = "dynet"
    PYG = "pyg"
    DGL = "dgl"
    JAVA = "java"


class Sample:
    def __init__(self, sample):
        self.id = get_field(sample, "id")
        self.target = float(get_field(sample, "target"))
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

        if self.value:
            self.value = json.loads(self.value)

        if self.weights is not None:
            self.weights = list(self.weights)

        if self.inputs is not None:
            self.inputs = list(self.inputs)


class Weight(object):
    def __init__(self, weight):
        self.index: int = get_field(weight, "index")
        self.name = get_field(weight, "name")
        self.dimensions = tuple(get_field(weight, "dimensions"))
        self.value = json.loads(get_field(weight, "value"))
        self.fixed = get_field(weight, "isFixed")

        if not isinstance(self.value, list):
            self.value = self.value

        if not self.dimensions:
            self.dimensions = (1,)

    @staticmethod
    def get_unit_weight() -> "Weight":
        weight = Weight.__new__(Weight)
        weight.index = 0
        weight.name = "unit"
        weight.dimensions = (1,)
        weight.value = 1.0
        weight.fixed = True

        return weight


class Builder:
    @staticmethod
    def get_builders(settings: Settings, sources: Optional[Sources]):
        namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.building
        builder = namespace.End2endTrainigBuilder(settings.settings, None if sources is None else sources.sources)
        nn_builder = builder.getEnd2endNNBuilder()

        return builder, nn_builder

    @staticmethod
    def build(source_pipeline, sources: Optional[Sources]):
        pipes_namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.pipes.specific
        serializer_pipe = pipes_namespace.NeuralSerializerPipe()

        source_pipeline.connectAfter(serializer_pipe)
        source_pipeline.execute(None if sources is None else sources.sources)
        return serializer_pipe.get()

    @staticmethod
    def from_sources(settings: Settings, backend: Backend, sources: Sources):
        builder, nn_builder = Builder.get_builders(settings, sources)

        if backend == Backend.JAVA:
            source_pipeline = nn_builder.buildPipeline()
            source_pipeline.execute(None if sources is None else sources.sources)
            java_model = source_pipeline.get()

            logic_samples = get_field(java_model, "s")
            neural_model = get_field(java_model, "r")
            samples = logic_samples.collect(get_neuralogic().java.util.stream.Collectors.toList())
            return neural_model, samples

        result = Builder.build(nn_builder.buildPipeline(), sources)

        serialized_weights = list(get_field(result, "r"))
        weights: List = [None] * len(serialized_weights)

        for x in serialized_weights:
            weight = Weight(x)
            weights[weight.index] = weight

        samples = [Sample(x) for x in stream_to_list(get_field(result, "s"))]

        return weights, samples

    @staticmethod
    def from_problem(parsed_template, logic_samples, backend: Backend, settings: Settings):
        builder, nn_builder = Builder.get_builders(settings, None)

        if backend == Backend.JAVA:
            source_pipeline = nn_builder.buildPipeline(parsed_template, logic_samples)
            source_pipeline.execute(None)
            java_model = source_pipeline.get()

            logic_samples = get_field(java_model, "s")
            neural_model = get_field(java_model, "r")

            samples = logic_samples.collect(get_neuralogic().java.util.stream.Collectors.toList())
            return neural_model, samples

        result = Builder.build(nn_builder.buildPipeline(parsed_template, logic_samples), None)

        dummy_weight = Weight.get_unit_weight()
        serialized_weights = list(get_field(result, "r"))
        weights: List = [dummy_weight] * len(serialized_weights)

        for x in serialized_weights:
            weight = Weight(x)

            if weight.index >= len(weights):
                weights.extend([dummy_weight] * (weight.index - len(weights) + 1))
            weights[weight.index] = weight

        samples = [Sample(x) for x in stream_to_list(get_field(result, "s"))]

        return weights, samples
