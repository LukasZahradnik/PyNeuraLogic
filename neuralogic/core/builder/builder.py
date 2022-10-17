from typing import List, Optional

import jpype
from halo._utils import get_environment

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

        @jpype.JImplements(jpype.JClass("java.util.function.IntConsumer"))
        class Callback:
            def __init__(self, spinner):
                self.state = 0
                self.spinner = spinner

            @jpype.JOverride
            def accept(self, count: int):
                self.state = count
                self.spinner.text = Builder._get_spinner_text(count)

        self._callback = Callback

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

    def from_sources(self, parsed_template, sources: Sources, progress: bool) -> List[RawSample]:
        if progress:
            return self._build_samples_with_progress(parsed_template, sources, None)
        return self._build_samples(parsed_template, sources, None)

    def from_logic_samples(self, parsed_template, logic_samples, progress: bool) -> List[RawSample]:
        if progress:
            return self._build_samples_with_progress(parsed_template, None, logic_samples)
        return self._build_samples(parsed_template, None, logic_samples)

    def _build_samples_with_progress(
        self, parsed_template, sources: Optional[Sources], logic_samples
    ) -> List[RawSample]:
        jupyter = get_environment() in ("ipython", "jupyter")

        if jupyter:
            from halo import HaloNotebook as Halo
        else:
            from halo import Halo

        self.spinner = Halo(text=self._get_spinner_text(0), spinner="dots")
        self.spinner.start()

        try:
            results = self._build_samples(parsed_template, sources, logic_samples, self._callback(self.spinner))

            self.spinner.succeed()

            if jupyter:
                self.spinner.start().succeed()
            return results
        except Exception as e:
            self.spinner.fail("Building failed")

            if jupyter:
                self.spinner.start().fail()
            raise e

    def _build_samples(
        self, parsed_template, sources: Optional[Sources], logic_samples, callback=None
    ) -> List[RawSample]:
        if logic_samples is None:
            source_pipeline = self.example_builder.buildPipeline(parsed_template, sources.sources, callback)
        else:
            source_pipeline = self.example_builder.buildPipeline(parsed_template, logic_samples, callback)

        source_pipeline.execute(None if sources is None else sources.sources)
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

    @staticmethod
    def _get_spinner_text(count: int) -> str:
        if count == 1:
            return f"Built {count} sample"
        return f"Built {count} samples"
