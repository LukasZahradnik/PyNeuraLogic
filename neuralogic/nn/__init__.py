from typing import Optional

from neuralogic.builder import Backend
from neuralogic.data import Dataset
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.model import Model
from neuralogic.settings import Settings


def get_neuralogic_layer(backend: Backend):
    if backend == Backend.DYNET:
        from neuralogic.nn.dynet import NeuraLogicLayer  # type: ignore

        return NeuraLogicLayer
    if backend == Backend.DGL:
        from neuralogic.nn.dgl import NeuraLogicLayer  # type: ignore

        return NeuraLogicLayer


def get_evaluator(backend: Backend, model: Model = None, dataset: Dataset = None, settings: Optional[Settings] = None):
    if model is None and dataset is None:
        raise NotImplementedError
    if model is not None and dataset is not None:
        raise NotImplementedError

    if settings is None:
        if model is not None:
            settings = model.java_factory.settings
        elif dataset is not None:
            settings = dataset.settings
        else:
            raise Exception

    if backend == Backend.DYNET:
        from neuralogic.nn.evaluators.dynet import DynetEvaluator

        return DynetEvaluator(model, dataset, settings)
    if backend == Backend.JAVA:
        from neuralogic.nn.evaluators.java import JavaEvaluator

        return JavaEvaluator(model, dataset, settings)
