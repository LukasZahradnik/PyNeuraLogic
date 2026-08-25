from typing import Any

import jpype

from neuralogic.core.constructs.function.function import TransformationFunction


class LeakyReLuTransformation(TransformationFunction):
    """A LeakyReLU whose negative slope can be given per rule.

    The backend's slope used to be one mutable static, global to the JVM, so no template could ask for a
    particular one and any comparison had to bend the other framework to `0.01` instead. PyG's `GATv2Conv`
    uses `0.2`, which is the case this exists for::

        Transformation.LEAKY_RELU               # the default, 0.01
        Transformation.LEAKY_RELU(0.2)          # this rule only
    """

    __slots__ = ("slope",)

    def __init__(self, name: str, *, namespace: str = "", slope: float | None = None):
        """
        Parameters
        ----------
        name : str
            The name of the transformation function.
        namespace : str
            The Java namespace of the function.
        slope : float, optional
            The negative slope. Default: None, meaning the backend's own default.
        """
        super().__init__(name, namespace=namespace)
        self.slope = slope

    def __call__(self, slope: float | None = None, *args: Any, **kwargs: Any) -> Any:
        """Returns a LeakyReLU with the given negative slope, leaving this one alone.

        Parameters
        ----------
        slope : float, optional
            The negative slope. Called with nothing, this is the plain function, so that
            ``Transformation.LEAKY_RELU`` keeps working wherever a function is called before use.

        Returns
        -------
        TransformationFunction
        """
        if slope is None:
            return TransformationFunction.__call__(self, *args, **kwargs)
        if args or kwargs:
            return TransformationFunction.__call__(
                LeakyReLuTransformation(self.name, namespace=self.namespace, slope=slope), *args, **kwargs
            )
        return LeakyReLuTransformation(self.name, namespace=self.namespace, slope=slope)

    def is_parametrized(self) -> bool:
        return self.slope is not None

    def get(self) -> Any:
        java_class = jpype.JClass("cz.cvut.fel.ida.algebra.functions.transformation.elementwise.LeakyReLu")
        if self.slope is None:
            return java_class()
        return java_class(float(self.slope))

    def __str__(self) -> str:
        if self.slope is None:
            return self.name
        return f"{self.name}({self.slope})"
