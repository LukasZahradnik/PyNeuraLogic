from typing import List, Optional

import jpype
from tqdm.autonotebook import tqdm

from neuralogic import is_initialized, initialize
from neuralogic.core.builder.components import NeuralSample
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

    def ground_from_sources(self, parsed_template, sources: Sources):
        return self._ground(parsed_template, sources, None)

    def ground_from_logic_samples(self, parsed_template, logic_samples):
        return self._ground(parsed_template, None, logic_samples)

    def _ground(self, parsed_template, sources: Optional[Sources], logic_samples) -> List[NeuralSample]:
        if sources is not None:
            ground_pipeline = self.example_builder.buildGroundings(parsed_template, sources.sources)
        else:
            logic_samples = jpype.java.util.ArrayList(logic_samples).stream()
            ground_pipeline = self.example_builder.buildGroundings(parsed_template, logic_samples)

        ground_pipeline.execute(None if sources is None else sources.sources)

        return ground_pipeline.get()

    def neuralize(self, groundings, progress: bool, length: Optional[int]) -> List[NeuralSample]:
        if not progress:
            return self._neuralize(groundings, None)
        with tqdm(total=length, desc="Building", unit=" samples", dynamic_ncols=True) as pbar:
            return self._neuralize(groundings, self._callback(pbar))

    def _neuralize(self, groundings, callback) -> List[NeuralSample]:
        neuralize_pipeline = self.example_builder.neuralize(groundings, callback)
        neuralize_pipeline.execute(None)

        samples = neuralize_pipeline.get()
        logic_samples = samples.collect(self.collectors.toList())

        return [NeuralSample(sample, None) for sample in logic_samples]

    def build_model(self, parsed_template, settings: SettingsProxy):
        neural_model = self.neural_model(parsed_template.getAllWeights(), settings.settings)

        return neural_model

    @staticmethod
    def get_builders(settings: SettingsProxy):
        builder = jpype.JClass("cz.cvut.fel.ida.pipelines.building.PythonBuilder")(settings.settings)

        return builder

    @staticmethod
    def _get_spinner_text(count: int) -> str:
        if count == 1:
            return f"Built {count} sample"
        return f"Built {count} samples"
