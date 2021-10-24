from typing import Optional

from neuralogic.core.builder import Backend
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core import Template
from neuralogic.core.settings import Settings


def get_neuralogic_layer(backend: Backend, native_backend_models: bool = False):
    if backend == Backend.DYNET:
        from neuralogic.nn.dynet import NeuraLogic  # type: ignore

        return NeuraLogic
    if backend == Backend.JAVA:
        from neuralogic.nn.java import NeuraLogic  # type: ignore

        return NeuraLogic
    if backend == Backend.PYG:
        from neuralogic.nn.native.torch import NeuraLogic

        return NeuraLogic
    raise NotImplementedError


def get_evaluator(
    template: Template,
    backend: Backend,
    settings: Optional[Settings] = None,
    *,
    native_backend_models=False,
):
    if settings is None:
        settings = Settings()

    if backend == Backend.DYNET:
        from neuralogic.nn.evaluators.dynet import DynetEvaluator

        return DynetEvaluator(template, settings)
    if backend == Backend.JAVA:
        from neuralogic.nn.evaluators.java import JavaEvaluator

        return JavaEvaluator(template, settings)
    if backend == Backend.PYG:
        from neuralogic.nn.evaluators.torch import TorchEvaluator

        return TorchEvaluator(template, settings)
