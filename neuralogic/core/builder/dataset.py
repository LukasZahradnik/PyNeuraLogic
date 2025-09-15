import jpype

from neuralogic.core.builder import Builder
from neuralogic.core.builder.components import NeuralSample, Grounding


class BuiltDataset:
    """BuiltDataset represents an already built dataset - that is, a dataset that has been grounded and neuralized."""

    __slots__ = "_samples", "_batch_size"

    def __init__(self, samples: list[NeuralSample], batch_size: int):
        self._samples = samples
        self._batch_size = batch_size

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        return self._samples[item]

    def __iter__(self):
        return iter(self._samples)


class GroundedDataset:
    """GroundedDataset represents grounded examples that are not neuralized yet."""

    __slots__ = "_groundings", "_groundings_list", "_builder"

    def __init__(self, groundings, builder: Builder):
        self._builder = builder
        self._groundings = groundings
        self._groundings_list = [Grounding(g) for g in self._groundings]

    def __getitem__(self, item) -> Grounding:
        return self._groundings_list[item]

    def __len__(self) -> int:
        return len(self._groundings_list)

    def __iter__(self):
        return iter(self._groundings_list)

    def neuralize(self, *, batch_size: int = 1, progress: bool = False) -> BuiltDataset:
        return BuiltDataset(self._builder.neuralize(self._groundings.stream(), progress, len(self)), batch_size)
