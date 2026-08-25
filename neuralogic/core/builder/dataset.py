from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jpype

from neuralogic.core.builder.components import Grounding, NeuralSample
from neuralogic.core.constructs.java_objects import ValueFactory

if TYPE_CHECKING:
    from neuralogic.core.builder import Builder


class IncompatibleKeys(NamedTuple):
    """What :meth:`BuiltDataset.load_state_dict` could not match up, in torch's shape."""

    missing_keys: list[str]
    unexpected_keys: list[str]


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

    def _example_parameters(self) -> dict:
        """Ground literal to the Java weight holding that fact's learnable value."""
        collector = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.ExampleParameters")
        java_samples = jpype.java.util.ArrayList([sample._java_sample for sample in self._samples])
        return {str(literal): weight for literal, weight in collector.of(java_samples).items()}

    def state_dict(self) -> dict:
        """The learnable values this dataset's own example facts carry, keyed by ground literal.

        A value written on an example fact and made learnable with ``learnable_facts=True`` is a real
        parameter - it trains - but it belongs to the *data*, not to the template, so it is not in the
        model's :meth:`~neuralogic.core.neural_module.NeuralModule.state_dict` and saving the model does not
        save it. This is where it is.

        The key is the ground literal (``emb(a)``) because nothing else survives a rebuild: the weight's
        index continues a counter that keeps running, so building the same dataset twice on one model gives
        the parameters indices ``1, 2`` and then ``5, 6``, and the generated weight name is ``w`` plus that
        index.

        Returns
        -------
        dict
            ``{"weights": {literal: value}}``, empty when the dataset was not built with
            ``learnable_facts=True``.
        """
        return {
            "weights": {
                literal: ValueFactory.from_java(weight.value)
                for literal, weight in self._example_parameters().items()
            }
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> IncompatibleKeys:
        """Sets this dataset's example-fact parameters from a :meth:`state_dict` shaped dictionary.

        With ``strict=False`` this is also how a parameter reaches a dataset it was not trained on - the
        embeddings learned for the training set, carried onto the test set for the constants the two share::

            test_data.load_state_dict(train_data.state_dict(), strict=False)

        A constant the training set never saw keeps whatever its example wrote, untrained, and comes back in
        ``missing_keys``; a trained parameter with no fact to attach to here comes back in
        ``unexpected_keys``. Both are ordinary in that use and neither is in the round trip of one dataset,
        which is why ``strict`` defaults to refusing them.

        Parameters
        ----------
        state_dict : dict
            ``{"weights": {literal: value}}``.
        strict : bool
            Whether the literals must match this dataset's exactly, as in torch. Default: True.

        Returns
        -------
        IncompatibleKeys
            ``missing_keys`` - this dataset's learnable facts the dictionary said nothing about - and
            ``unexpected_keys`` - literals in the dictionary this dataset has no learnable fact for.

        Raises
        ------
        ValueError
            Under ``strict``, if either list is non-empty: the saved parameters and the data have drifted
            apart, and carrying on would leave a model half restored. Always, if two literals that share one
            weight are given different values - facts share a weight when the example named it, and then
            only one of the two values could survive, which one being an accident of iteration order.
        """
        parameters = self._example_parameters()
        weights = state_dict["weights"] if "weights" in state_dict else state_dict

        unexpected = [literal for literal in weights if literal not in parameters]
        missing = [literal for literal in parameters if literal not in weights]

        if strict and (unexpected or missing):
            raise ValueError(
                f"parameters do not match this dataset: {len(missing)} missing {missing[:3]}, "
                f"{len(unexpected)} unexpected {unexpected[:3]}. Pass strict=False to carry over only what "
                f"the two have in common, which is what moving trained parameters to another dataset means"
            )

        shared: dict[int, tuple[str, Any]] = {}
        for literal, value in weights.items():
            if literal in unexpected:
                continue
            index = int(parameters[literal].index)
            seen = shared.get(index)
            if seen is not None and seen[1] != value:
                raise ValueError(
                    f"{seen[0]} and {literal} share one weight, so they cannot be given different values"
                )
            shared[index] = (literal, value)

        for literal, value in weights.items():
            if literal not in unexpected:
                _set_value(parameters[literal].value, value)

        return IncompatibleKeys(missing, unexpected)


def _set_value(weight_value, value) -> None:
    """The same flattening NeuralModule uses, kept here so a dataset does not have to import the model."""
    if isinstance(value, (float, int)):
        weight_value.set(0, float(value))
        return
    if isinstance(value[0], (float, int)):
        for i, val in enumerate(value):
            weight_value.set(i, float(val))
        return

    cols = len(value[0])
    for i, values in enumerate(value):
        for j, val in enumerate(values):
            weight_value.set(i * cols + j, float(val))


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
