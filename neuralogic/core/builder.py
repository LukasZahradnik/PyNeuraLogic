from neuralogic import get_neuralogic, get_gateway
from neuralogic.core.settings import Settings
from neuralogic.core.sources import Sources

import json
import numpy as np
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


class Builder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.example_builder = Builder.get_builders(settings)
        self.builder = Builder.get_builders(settings)

    def build_template_from_file(self, settings: Settings, filename: str):
        args = [
            "-t",
            filename,
            "-q",
            filename,
        ]

        sources = Sources.from_args(args, settings)
        template = self.builder.buildTemplate(sources.sources)

        return template

    def from_sources(self, parsed_template, sources: Sources, backend: Backend):
        if backend == Backend.JAVA:
            source_pipeline = self.example_builder.buildPipeline(parsed_template, sources.sources)
            source_pipeline.execute(None if sources is None else sources.sources)
            java_model = source_pipeline.get()

            logic_samples = get_field(java_model, "s")
            return logic_samples.collect(get_neuralogic().java.util.stream.Collectors.toList())

        result = Builder.build(self.example_builder.buildPipeline(parsed_template, sources.sources), sources)
        return [Sample(x) for x in stream_to_list(get_field(result, "s"))]

    def from_logic_samples(self, parsed_template, logic_samples, backend: Backend):
        if backend == Backend.JAVA:
            source_pipeline = self.example_builder.buildPipeline(parsed_template, logic_samples)
            source_pipeline.execute(None)
            java_model = source_pipeline.get()

            logic_samples = get_field(java_model, "s")
            return logic_samples.collect(get_neuralogic().java.util.stream.Collectors.toList())

        result = Builder.build(self.example_builder.buildPipeline(parsed_template, logic_samples), None)
        return [Sample(x) for x in stream_to_list(get_field(result, "s"))]

    def build_model(self, parsed_template, backend: Backend, settings: Settings):
        namespace = get_neuralogic().cz.cvut.fel.ida.neural.networks.computation.training

        neural_model = namespace.NeuralModel(parsed_template.getAllWeights(), settings.settings)

        if backend == Backend.JAVA:
            return neural_model

        dummy_weight = Weight.get_unit_weight()
        weights: List = [dummy_weight] * len(parsed_template.getAllWeights())

        for x in parsed_template.getAllWeights():
            weight = Weight(x)

            if weight.index >= len(weights):
                weights.extend([dummy_weight] * (weight.index - len(weights) + 1))
            weights[weight.index] = weight
        return weights

    @staticmethod
    def get_builders(settings: Settings):
        namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.building
        builder = namespace.PythonBuilder(settings.settings)

        return builder

    @staticmethod
    def build(source_pipeline, sources: Optional[Sources]):
        pipes_namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.pipes.specific
        serializer_pipe = pipes_namespace.NeuralSerializerPipe()

        source_pipeline.connectAfter(serializer_pipe)
        source_pipeline.execute(None if sources is None else sources.sources)
        return serializer_pipe.get()
