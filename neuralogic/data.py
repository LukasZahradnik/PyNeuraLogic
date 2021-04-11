from typing import Optional, Union, List
from pathlib import Path
import os
from neuralogic.builder import Weight, Sample, Builder
from neuralogic.settings import Settings
from neuralogic.sources import Sources


PathType = Optional[Union[Path, str]]


class Dataset:
    def __init__(
        self,
        source_dir: PathType = None,
        template: PathType = None,
        examples: PathType = None,
        queries: PathType = None,
    ):
        self.source_dir = source_dir

        self.template = template
        self.examples = examples
        self.queries = queries

        self.loaded = False
        self.__weights: List[Weight] = []
        self.__samples: List[Sample] = []

        self.settings: Optional[Settings] = None
        self.sources: Optional[Sources] = None

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

        settings = Settings()
        sources = Sources.from_args(args, settings)
        weights, samples = Builder.from_sources(settings, sources)

        self.__weights = weights
        self.__samples = samples

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


base_path = os.path.abspath(os.path.dirname(__file__))

XOR = Dataset(source_dir=os.path.join(base_path, "..", "dataset", "simple", "xor", "naive"))
XOR_Vectorized = Dataset(source_dir=os.path.join(base_path, "..", "dataset", "simple", "xor", "vectorized"))
Mutagenesis = Dataset(source_dir=os.path.join(base_path, "..", "dataset", "molecules", "mutagenesis"))
