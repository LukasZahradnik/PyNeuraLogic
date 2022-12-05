from typing import Union, Tuple

import jpype

from neuralogic.core.constructs.function.function import Transformation


class Slice(Transformation):
    __slots__ = ("rows", "cols")

    def __init__(
        self,
        name: str,
        *,
        rows: Union[type(Ellipsis), Tuple[int, int]] = ...,
        cols: Union[type(Ellipsis), Tuple[int, int]] = ...,
    ):
        super().__init__(name)

        if cols is not Ellipsis:
            cols = [int(x) for x in cols]

        if rows is not Ellipsis:
            rows = [int(x) for x in rows]

        self.rows = rows
        self.cols = cols

    def __call__(
        self,
        entity=None,
        *,
        rows: Union[type(Ellipsis), Tuple[int, int]] = ...,
        cols: Union[type(Ellipsis), Tuple[int, int]] = ...,
    ):
        slice = Slice(self.name, rows=rows, cols=cols)
        return Transformation.__call__(slice, entity)

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
