from typing import Union, Tuple, Optional

import jpype

from neuralogic.core.constructs.function.function import TransformationFunction


class Reshape(TransformationFunction):
    __slots__ = ("shape",)

    def __init__(
        self,
        name: str,
        *,
        shape: Union[None, Tuple[int, int], int] = None,
    ):
        super().__init__(name)

        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape

    def __call__(
        self,
        relation: Optional = None,
        *,
        shape: Union[None, Tuple[int, int], int] = None,
    ):
        reshape = Reshape(self.name, shape=shape)
        return TransformationFunction.__call__(reshape, relation)

    def is_parametrized(self) -> bool:
        return True

    def get(self):
        shape = None if self.shape is None else list(self.shape)

        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.transformation.joint.Reshape")(shape)

    def wrap(self, content: str) -> str:
        return f"reshape({content}, shape={self.shape})"

    def __str__(self):
        return f"reshape(shape={self.shape})"
