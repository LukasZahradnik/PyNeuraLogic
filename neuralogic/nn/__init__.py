from typing import Optional, Tuple

from neuralogic.core.builder import Backend
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core import Problem
from neuralogic.core.settings import Settings


def get_neuralogic_layer(backend: Backend):
    if backend == Backend.DYNET:
        from neuralogic.nn.dynet import NeuraLogic  # type: ignore

        return NeuraLogic
    if backend == Backend.DGL:
        from neuralogic.nn.dgl import NeuraLogicLayer  # type: ignore

        return NeuraLogicLayer
    if backend == Backend.JAVA:
        from neuralogic.nn.java import NeuraLogic  # type: ignore

        return NeuraLogic


def get_evaluator(
    backend: Backend,
    problem: Problem,
    settings: Optional[Settings] = None,
):
    if settings is None:
        if problem is not None:
            settings = problem.java_factory.settings
        else:
            settings = Settings()

    if backend == Backend.DYNET:
        from neuralogic.nn.evaluators.dynet import DynetEvaluator

        return DynetEvaluator(problem, settings)
    if backend == Backend.JAVA:
        from neuralogic.nn.evaluators.java import JavaEvaluator

        return JavaEvaluator(problem, settings)
