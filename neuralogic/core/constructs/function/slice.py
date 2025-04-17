from typing import Tuple
from types import EllipsisType

import jpype

from neuralogic.core.constructs.function.function import TransformationFunction


class Slice(TransformationFunction):
    __slots__ = ("rows", "cols")

    def __init__(
        self,
        name: str,
        *,
        rows: EllipsisType | Tuple[int, int] = ...,
        cols: EllipsisType | Tuple[int, int] = ...,
    ):
        super().__init__(name)

        self.cols = [int(x) for x in cols] if cols is not Ellipsis else Ellipsis
        self.rows = [int(x) for x in rows] if rows is not Ellipsis else Ellipsis

    def __call__(
        self,
        relation=None,
        *,
        rows: EllipsisType | Tuple[int, int] = ...,
        cols: EllipsisType | Tuple[int, int] = ...,
    ):
        slice = Slice(self.name, rows=rows, cols=cols)
        return TransformationFunction.__call__(slice, relation)

    def is_parametrized(self) -> bool:
        return True

    def get(self):
        cols = None if self.cols is Ellipsis else self.cols
        rows = None if self.rows is Ellipsis else self.rows

        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.transformation.joint.Slice")(rows, cols)

    def wrap(self, content: str) -> str:
        rows = "..." if self.rows is Ellipsis or self.rows is None else self.rows
        cols = "..." if self.cols is Ellipsis or self.cols is None else self.cols

        return f"slice({content}, rows={rows}, cols={cols})"

    def __str__(self):
        rows = "..." if self.rows is Ellipsis or self.rows is None else self.rows
        cols = "..." if self.cols is Ellipsis or self.cols is None else self.cols

        return f"slice(rows={rows}, cols={cols})"
