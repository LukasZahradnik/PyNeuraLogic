import jpype
from typing import Any
from tqdm.autonotebook import tqdm

from neuralogic.setup import is_initialized, initialize
from neuralogic.core.builder.components import NeuralSample
from neuralogic.core.settings import SettingsProxy
from neuralogic.core.sources import Sources


def stream_to_list(stream: Any) -> list:
    """
    Converts a Java stream to a Python list.

    Parameters
    ----------
    stream : Any
        The Java stream to convert.

    Returns
    -------
    list
        The converted Python list.
    """
    return list(stream.collect(jpype.JClass("java.util.stream.Collectors").toList()))


class Builder:
    """
    Builder is responsible for grounding, neuralizing, and building models from models and sources.
    """

    def __init__(self, settings: SettingsProxy):
        """
        Parameters
        ----------
        settings : SettingsProxy
            The settings proxy for the builder.
        """
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

    def build_model_from_file(self, settings: SettingsProxy, filename: str) -> Any:
        """
        Builds a model from a file.

        Parameters
        ----------
        settings : SettingsProxy
            The settings proxy.
        filename : str
            The path to the model file.

        Returns
        -------
        Any
            The built model.
        """
        args = [
            "-t",
            filename,
            "-q",
            filename,
        ]

        sources = Sources.from_args(args, settings)
        model = self.builder.buildModel(sources.sources)

        return model

    def ground_from_sources(self, parsed_model: Any, sources: Sources, progress: bool) -> Any:
        """
        Grounds the model from the provided sources.

        Parameters
        ----------
        parsed_model : Any
            The parsed model.
        sources : Sources
            The logic sources.
        progress : bool
            Whether to show progress.

        Returns
        -------
        Any
            The grounded logic samples.
        """
        if not progress:
            return self._ground(parsed_model, sources, None, None)
        with tqdm(total=None, desc="Grounding", unit=" samples", dynamic_ncols=True) as pbar:
            return self._ground(parsed_model, sources, None, self._callback(pbar))

    def ground_from_logic_samples(self, parsed_model: Any, logic_samples: list[Any], progress: bool) -> Any:
        """
        Grounds the model from the provided logic samples.

        Parameters
        ----------
        parsed_model : Any
            The parsed model.
        logic_samples : list[Any]
            The logic samples.
        progress : bool
            Whether to show progress.

        Returns
        -------
        Any
            The grounded logic samples.
        """
        if not progress:
            return self._ground(parsed_model, None, logic_samples, None)
        with tqdm(total=len(logic_samples), desc="Grounding", unit=" samples", dynamic_ncols=True) as pbar:
            return self._ground(parsed_model, None, logic_samples, self._callback(pbar))

    def _ground(self, parsed_model: Any, sources: Sources | None, logic_samples: list[Any] | None, callback: Any) -> Any:
        if sources is not None:
            ground_pipeline = self.example_builder.buildGroundings(parsed_model, sources.sources, callback)
        else:
            logic_samples = jpype.java.util.ArrayList(logic_samples).stream()
            ground_pipeline = self.example_builder.buildGroundings(parsed_model, logic_samples, callback)

        ground_pipeline.execute(None if sources is None else sources.sources)
        groundings = ground_pipeline.get()

        if callback is not None:
            return groundings.collect(self.collectors.toList())
        return groundings

    def neuralize(self, groundings, progress: bool, length: int | None) -> list[NeuralSample]:
        """
        Neuralizes the grounding samples.

        Parameters
        ----------
        groundings : Any
            The logic groundings to neuralize.
        progress : bool
            Whether to show progress.
        length : int, optional
            The total number of groundings. Default: None.

        Returns
        -------
        list[NeuralSample]
            The neuralized samples.
        """
        if not progress:
            return self._neuralize(groundings, None)
        with tqdm(total=length, desc="Building", unit=" samples", dynamic_ncols=True) as pbar:
            return self._neuralize(groundings, self._callback(pbar))

    def _neuralize(self, groundings: Any, callback: Any) -> list[NeuralSample]:
        neuralize_pipeline = self.example_builder.neuralize(groundings, callback)
        neuralize_pipeline.execute(None)
        logic_samples = neuralize_pipeline.get().collect(self.collectors.toList())

        return [NeuralSample(sample) for sample in logic_samples]

    def build_model(self, parsed_model: Any, settings: SettingsProxy) -> Any:
        """
        Builds a neural model from the parsed model.

        Parameters
        ----------
        parsed_model : Any
            The parsed model.
        settings : SettingsProxy
            The settings proxy.

        Returns
        -------
        Any
            The built neural model.
        """
        neural_model = self.neural_model(parsed_model.getAllWeights(), settings.settings)

        return neural_model

    @staticmethod
    def get_builders(settings: SettingsProxy) -> Any:
        builder = jpype.JClass("cz.cvut.fel.ida.pipelines.building.PythonBuilder")(settings.settings)

        return builder

    @staticmethod
    def _get_spinner_text(count: int) -> str:
        if count == 1:
            return f"Built {count} sample"
        return f"Built {count} samples"
