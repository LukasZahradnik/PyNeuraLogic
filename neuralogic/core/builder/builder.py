from typing import List, Optional

import jpype
from tqdm.autonotebook import tqdm

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
            def __init__(self, progress_bar):
                self.state = 0
                self.progress_bar = progress_bar

            @jpype.JOverride
            def accept(self, count: int):
                self.state = max(count, self.state)
                if not self.progress_bar.disable:
                    self.progress_bar.update(1)

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
        total = None if logic_samples is None else len(logic_samples)

        with tqdm(total=total, desc="Building", unit=" samples", dynamic_ncols=True) as pbar:
            return self._build_samples(parsed_template, sources, logic_samples, self._callback(pbar))

    def _build_samples(
        self, parsed_template, sources: Optional[Sources], logic_samples, callback=None
    ) -> List[RawSample]:
        if sources is not None:
            source_pipeline = self.example_builder.buildPipeline(parsed_template, sources.sources, callback)
        else:
            logic_samples = jpype.java.util.ArrayList(logic_samples).stream()
            source_pipeline = self.example_builder.buildPipeline(parsed_template, logic_samples, callback)

        source_pipeline.execute(None if sources is None else sources.sources)
        java_model = source_pipeline.get()

        groundings = java_model.r.collect(self.collectors.toList())
        logic_samples = java_model.s.collect(self.collectors.toList())

        return [RawSample(sample, grounding) for sample, grounding in zip(logic_samples, groundings)]

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
