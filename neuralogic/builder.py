from . import get_neuralogic
from .helpers import stream_to_list
from .settings import Settings
from .sources import Sources
import json
from typing import List
from py4j.java_gateway import get_field


OFFSET = 1


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


class Weight(object):
    def __init__(self, weight):
        self.index = get_field(weight, "index")
        self.name = get_field(weight, "name")
        self.dimensions = tuple(get_field(weight, "dimensions"))
        self.value = json.loads(get_field(weight, "value"))
        self.fixed = get_field(weight, "isFixed")

        if not isinstance(self.value, list):
            self.value = [self.value]

        if not self.dimensions:
            self.dimensions = (1,)


class Model:
    def __init__(self, weights: List[Weight], samples: List[Sample]):
        self.weights = weights
        self.samples = samples

    @staticmethod
    def from_neuralogic(settings: Settings, sources: Sources) -> "Model":
        namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.building
        pipes_namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.pipes.specific

        builder = namespace.End2endTrainigBuilder(settings.settings, sources.sources)
        nn_builder = builder.getEnd2endNNBuilder()

        pipeline = nn_builder.buildPipeline()
        serializer_pipe = pipes_namespace.NeuralSerializerPipe()

        pipeline.connectAfter(serializer_pipe)
        pipeline.execute(sources.sources)
        result = serializer_pipe.get()

        weights = sorted([Weight(x) for x in get_field(result, "r")], key=lambda a: a.index)
        sample = [Sample(x) for x in stream_to_list(get_field(result, "s"))]

        return Model(weights, sample)
