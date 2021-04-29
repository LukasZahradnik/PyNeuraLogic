from typing import Optional, Tuple

from neuralogic.core.builder import Backend
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core import Problem
from neuralogic.core.settings import Settings
from neuralogic.utils.data import Dataset


def get_neuralogic_layer(backend: Backend):
    if backend == Backend.DYNET:
        from neuralogic.nn.dynet import NeuraLogicLayer  # type: ignore

        return NeuraLogicLayer
    if backend == Backend.DGL:
        from neuralogic.nn.dgl import NeuraLogicLayer  # type: ignore

        return NeuraLogicLayer
    if backend == Backend.JAVA:
        from neuralogic.nn.java import NeuraLogicLayer  # type: ignore

        return NeuraLogicLayer


def get_evaluator(
    backend: Backend,
    model_and_dataset=None,
    problem: Problem = None,
    settings: Optional[Settings] = None,
):
    model, dataset = None, None

    if problem is None and model_and_dataset is None:
        raise NotImplementedError
    if problem is not None and model_and_dataset is not None:
        raise NotImplementedError

    if settings is None:
        if problem is not None:
            settings = problem.java_factory.settings
        elif model_and_dataset is not None:
            model, dataset = model_and_dataset

            settings = model.settings
        else:
            raise Exception

    if backend == Backend.DYNET:
        from neuralogic.nn.evaluators.dynet import DynetEvaluator

        return DynetEvaluator(problem, model, dataset, settings)
    if backend == Backend.JAVA:
        from neuralogic.nn.evaluators.java import JavaEvaluator

        return JavaEvaluator(problem, model, dataset, settings)
