from neuralogic.nn.evaluator.java import JavaEvaluator
from neuralogic.nn.java import NeuraLogic  # type: ignore


def get_neuralogic_layer():
    return NeuraLogic


def get_evaluator(
    template,
    settings=None,
):
    from neuralogic.core.settings import Settings

    if settings is None:
        settings = Settings()
    return JavaEvaluator(template, settings)
