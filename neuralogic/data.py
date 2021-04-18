from typing import Optional, Union, List
from pathlib import Path
import os

from py4j.java_gateway import get_field

from neuralogic import get_neuralogic
from neuralogic.builder import Weight, Sample, Builder, Backend
from neuralogic.settings import Settings
from neuralogic.sources import Sources


PathType = Optional[Union[Path, str]]


class Dataset:
    def __init__(
        self,
        settings: Settings,
        backend: Backend,
        source_dir: PathType = None,
        template: PathType = None,
        examples: PathType = None,
        queries: PathType = None,
    ):
        self.source_dir = source_dir
        self.backend = backend

        self.template = template
        self.examples = examples
        self.queries = queries

        self.loaded = False
        self.__weights: List[Weight] = []
        self.__samples: List[Sample] = []

        self.__neural_model = None

        self.settings = settings
        self.sources: Optional[Sources] = None

        self.java_model = None

    def load(self, args: Optional[List] = None):
        self.loaded = True

        args = [] if args is None else args

        if self.source_dir is not None:
            args.extend(["-sd", str(self.source_dir)])
        if self.template is not None:
            args.extend(["-t", str(self.template)])
        if self.queries is not None:
            args.extend(["-q", str(self.queries)])
        if self.examples is not None:
            args.extend(["-e", str(self.examples)])

        sources = Sources.from_args(args, self.settings)

        if self.backend == Backend.JAVA:
            java_model = Builder.from_sources(self.settings, self.backend, sources)

            logic_samples = get_field(java_model, "s")
            self.__neural_model = get_field(java_model, "r")
            self.__samples = logic_samples.collect(get_neuralogic().java.util.stream.Collectors.toList())
            return

        weights, samples = Builder.from_sources(self.settings, self.backend, sources)

        self.__weights = weights
        self.__samples = samples

    @property
    def neural_model(self):
        if not self.loaded:
            self.load()
        return self.__neural_model

    @property
    def samples(self) -> List[Sample]:
        if not self.loaded:
            self.load()
        return self.__samples

    @property
    def weights(self) -> List[Weight]:
        if not self.loaded:
            self.load()
        return self.__weights


base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "dataset")


def XOR(settings: Settings, backend: Backend) -> Dataset:
    return Dataset(settings, backend, source_dir=os.path.join(base_path, "simple", "xor", "naive"))


def XOR_Vectorized(settings: Settings, backend: Backend) -> Dataset:
    return Dataset(settings, backend, source_dir=os.path.join(base_path, "simple", "xor", "vectorized"))


def Mutagenesis(settings: Settings, backend: Backend) -> Dataset:
    return Dataset(settings, backend, source_dir=os.path.join(base_path, "molecules", "mutagenesis"))
