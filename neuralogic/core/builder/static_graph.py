from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuralogic.core.builder.components import NeuralSample
from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core.constructs.rule import Rule

if TYPE_CHECKING:
    from neuralogic.core.builder.dataset_builder import DatasetBuilder
    from neuralogic.core.settings import SettingsProxy


class StaticGraphDataset:
    """A dataset that grounds and neuralizes only the first sample, then reuses
    the same neural graph for all subsequent samples by updating fact values
    via ``NeuralSample.set_fact_value``.

    This is useful when all samples share identical structure (same predicates,
    same term groundings) and differ only in fact values.

    Parameters
    ----------
    static_sample : NeuralSample
        The single neuralized sample built from the first example.
    fact_mappings : list[list[tuple[BaseRelation, float]]]
        For each original sample, a list of ``(fact, value)`` pairs to set
        before evaluating that sample.
    batch_size : int
        The batch size.
    """

    __slots__ = ("static_sample", "_fact_mappings", "_batch_size", "_targets")

    def __init__(
        self,
        static_sample: NeuralSample,
        fact_mappings: list[list[tuple[BaseRelation, Any]]],
        batch_size: int = 1,
        targets: list[Any] | None = None,
    ):
        self.static_sample = static_sample
        self._fact_mappings = fact_mappings
        self._batch_size = batch_size
        self._targets = [None] * len(fact_mappings) if targets is None else targets

    @property
    def fact_mappings(self) -> list[list[tuple[BaseRelation, Any]]]:
        """Return the per-sample fact→value mappings."""
        return self._fact_mappings

    @property
    def targets(self) -> list[Any]:
        """Each sample's own target, in the same order as :attr:`fact_mappings`.

        The static sample carries the *first* sample's target, so without these every sample would be
        fitted to that one label.
        """
        return self._targets

    def __len__(self) -> int:
        return len(self._fact_mappings)

    def apply_mapping(self, index: int) -> int:
        """Apply the fact values for the sample at the given index.

        Parameters
        ----------
        index : int
            The sample index.

        Returns
        -------
        int
            Number of facts updated.
        """
        count = 0
        for fact, value in self._fact_mappings[index]:
            self.static_sample.set_fact_value(fact, value)
            count += 1
        return count


def _extract_fact_value(entry: Any) -> Any | None:
    """Extract the value from a fact entry, as written.

    The value is handed on unchanged rather than reduced to a number, because a fact's value need not be a
    scalar: a fixed-width feature vector per example is the ordinary case for the kind of problem a static
    graph is for. ``ValueFactory.get_value`` converts whatever this returns, and keeping only the first
    element of a vector made the write fail with "scalar increment by vector".

    Parameters
    ----------
    entry : Any
        A ``BaseRelation`` or ``WeightedRelation`` representing a fact.

    Returns
    -------
    Any or None
        The fact's value - a number, or a sequence for a vector-valued fact - or ``None`` if the entry is a
        Rule (should be skipped).
    """
    if isinstance(entry, Rule):
        return None
    if isinstance(entry, WeightedRelation):
        weight = entry.weight
        if isinstance(weight, (tuple, list)) and len(weight) == 0:
            return 1.0
        return weight
    # Default: unweighted facts have value 1.0
    return 1.0


def _build_fact_mappings(
    dataset: Any,
) -> list[list[tuple[BaseRelation, float]]]:
    """Extract fact→value mappings from all samples in a Dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset with samples.

    Returns
    -------
    list[list[tuple[BaseRelation, float]]]
        For each sample, a list of ``(fact, value)`` pairs.
    """
    mappings: list[list[tuple[BaseRelation, float]]] = []

    for sample in dataset.samples:
        sample_mapping: list[tuple[BaseRelation, float]] = []
        for entry in sample.example:
            value = _extract_fact_value(entry)
            if value is not None:
                sample_mapping.append((entry, value))
        mappings.append(sample_mapping)

    return mappings


def build_static_graph_dataset(
    dataset_builder: DatasetBuilder,
    dataset: Any,
    settings: SettingsProxy,
    *,
    batch_size: int = 1,
    learnable_facts: bool = False,
    progress: bool = False,
) -> StaticGraphDataset:
    """Build a ``StaticGraphDataset`` from a logic ``Dataset``.

    Only the **first** sample is grounded and neuralized.  For every sample
    (including the first), a fact→value mapping is stored so that the caller
    can update the static sample before each learning/evaluation step.

    Parameters
    ----------
    dataset_builder : DatasetBuilder
    dataset : Dataset
        The dataset with one or more samples.
    settings : SettingsProxy
    batch_size : int
        Batch size (default 1).
    learnable_facts : bool
        Whether facts are learnable (default False).
    progress : bool
        Whether to show a progress bar (default False).

    Returns
    -------
    StaticGraphDataset
    """
    from neuralogic.dataset import Dataset as LogicDataset

    if not isinstance(dataset, LogicDataset) or len(dataset.samples) == 0:
        raise ValueError("Static graph requires a Dataset with at least one sample")

    if batch_size != 1:
        # learnSample is called one sample at a time, so a batch above one would be accepted and then
        # quietly ignored - the updates would never be accumulated into a single step
        raise NotImplementedError(
            f"a static graph trains one sample at a time, so batch_size must be 1, got {batch_size}"
        )

    for index, sample in enumerate(dataset.samples):
        if isinstance(sample.query, list):
            raise NotImplementedError(
                f"sample {index} has several queries; a static graph reuses one neural sample, and that "
                f"holds one query"
            )

    # Build fact mappings for all samples
    fact_mappings = _build_fact_mappings(dataset)
    targets = [
        _extract_fact_value(sample.query) if sample.query is not None else None
        for sample in dataset.samples
    ]

    # Build only the first sample
    first_sample = dataset.samples[0]
    first_dataset = LogicDataset([first_sample])

    built = dataset_builder.build_dataset(
        first_dataset,
        settings,
        batch_size=batch_size,
        learnable_facts=learnable_facts,
        progress=progress,
    )

    return StaticGraphDataset(
        static_sample=built[0],
        fact_mappings=fact_mappings,
        batch_size=batch_size,
        targets=targets,
    )
