from neuralogic import get_neuralogic
from neuralogic.helpers import stream_to_list
from neuralogic.settings import Settings
from neuralogic.sources import Sources
import json
from typing import List, Tuple
from py4j.java_gateway import get_field


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
            self.value = float(self.value)


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


class Model:
    @staticmethod
    def from_neuralogic(settings: Settings, sources: Sources) -> Tuple[List[Weight], List[Sample]]:
        namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.building
        pipes_namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.pipes.specific

        builder = namespace.End2endTrainigBuilder(settings.settings, sources.sources)
        nn_builder = builder.getEnd2endNNBuilder()

        pipeline = nn_builder.buildPipeline()
        serializer_pipe = pipes_namespace.NeuralSerializerPipe()

        pipeline.connectAfter(serializer_pipe)
        pipeline.execute(sources.sources)
        result = serializer_pipe.get()

        serialized_weights = list(get_field(result, "r"))
        weights: List = [None] * len(serialized_weights)

        for x in serialized_weights:
            weight = Weight(x)
            weights[weight.index] = weight

        sample = [Sample(x) for x in stream_to_list(get_field(result, "s"))]

        return weights, sample
