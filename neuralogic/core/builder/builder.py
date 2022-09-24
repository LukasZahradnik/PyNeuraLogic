from typing import List

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.builder.components import Sample, RawSample
from neuralogic.core.settings import SettingsProxy
from neuralogic.core.sources import Sources


def stream_to_list(stream) -> List:
    return list(stream.collect(jpype.JClass("java.util.stream.Collectors").toList()))


class Builder:
    def __init__(self, settings: SettingsProxy):
        if not is_initialized():
            initialize()

        self.settings = settings
        self.example_builder = Builder.get_builders(settings)
        self.builder = Builder.get_builders(settings)

        self.neural_model = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.NeuralModel")
        self.collectors = jpype.JClass("java.util.stream.Collectors")

    def build_template_from_file(self, settings: SettingsProxy, filename: str):
        args = [
            "-t",
            filename,
            "-q",
            filename,
        ]

        sources = Sources.from_args(args, settings)
        template = self.builder.buildTemplate(sources.sources)

        return template

    def from_sources(self, parsed_template, sources: Sources):
        source_pipeline = self.example_builder.buildPipeline(parsed_template, sources.sources)
        source_pipeline.execute(None if sources is None else sources.sources)
        java_model = source_pipeline.get()

        logic_samples = java_model.s
        logic_samples = logic_samples.collect(self.collectors.toList())

        return [RawSample(sample) for sample in logic_samples]

    def from_logic_samples(
        self,
        parsed_template,
        logic_samples,
    ):
        source_pipeline = self.example_builder.buildPipeline(parsed_template, logic_samples)
        source_pipeline.execute(None)
        java_model = source_pipeline.get()

        logic_samples = java_model.s
        logic_samples = logic_samples.collect(self.collectors.toList())

        return [RawSample(sample) for sample in logic_samples]

    def build_model(self, parsed_template, settings: SettingsProxy):
        neural_model = self.neural_model(parsed_template.getAllWeights(), settings.settings)

        return neural_model

    @staticmethod
    def get_builders(settings: SettingsProxy):
        builder = jpype.JClass("cz.cvut.fel.ida.pipelines.building.PythonBuilder")(settings.settings)

        return builder

    @staticmethod
    def build(samples):
        serializer = jpype.JClass("cz.cvut.fel.ida.neural.networks.structure.export.NeuralSerializer")()
        super_detailed_format = jpype.JClass("cz.cvut.fel.ida.setup.Settings").superDetailedNumberFormat
        serializer.numberFormat = super_detailed_format

        return [Sample(serializer.serialize(sample), sample) for sample in samples]
