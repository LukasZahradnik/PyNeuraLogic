from typing import Any

import jpype

from neuralogic.core.constructs.function.function import TransformationFunction


class Reshape(TransformationFunction):
    """
    Represents a reshape transformation function that changes the shape of a tensor.
    """

    __slots__ = ("shape",)

    def __init__(
        self,
        name: str,
        *,
        shape: tuple[int, int] | int | None = None,
    ):
        """
        Parameters
        ----------
        name : str
            The name of the function.
        shape : Union[Tuple[int, int], int], optional
            The target shape. Default: None.
        """
        super().__init__(name)

        self.shape = (shape,) if isinstance(shape, int) else shape

    def __call__(
        self,
        relation: Any | None = None,
        *,
        shape: tuple[int, int] | int | None = None,
    ) -> Any:
        """
        Creates a new Reshape instance with the provided shape and applies it to the relation.

        Parameters
        ----------
        relation : Any, optional
            The relation to apply the reshape to. Default: None.
        shape : tuple[int, int] | int | None, optional
            The target shape. Default: None.

        Returns
        -------
        TransformationFunction
            The new Reshape instance (attached to the relation if provided).
        """
        reshape = Reshape(self.name, shape=shape)
        return TransformationFunction.__call__(reshape, relation)

    def is_parametrized(self) -> bool:
        return True

    def get(self) -> Any:
        shape = None if self.shape is None else list(self.shape)

        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.transformation.joint.Reshape")(shape)

    def wrap(self, content: str) -> str:
        return f"reshape({content}, shape={self.shape})"

    def __str__(self) -> str:
        return f"reshape(shape={self.shape})"
