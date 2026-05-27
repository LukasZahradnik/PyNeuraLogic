from types import EllipsisType
from typing import Any

import jpype

from neuralogic.core.constructs.function.function import TransformationFunction


class Slice(TransformationFunction):
    """
    Represents a slice transformation function that extracts a sub-tensor from a tensor.
    """

    __slots__ = ("rows", "cols")

    def __init__(
        self,
        name: str,
        *,
        rows: EllipsisType | tuple[int, int] = ...,
        cols: EllipsisType | tuple[int, int] = ...,
    ):
        """
        Parameters
        ----------
        name : str
            The name of the function.
        rows : EllipsisType | tuple[int, int], optional
            The row range to slice. Default: Ellipsis (all rows).
        cols : EllipsisType | tuple[int, int], optional
            The column range to slice. Default: Ellipsis (all columns).
        """
        super().__init__(name)

        self.cols = [int(x) for x in cols] if cols is not Ellipsis else Ellipsis
        self.rows = [int(x) for x in rows] if rows is not Ellipsis else Ellipsis

    def __call__(
        self,
        relation: Any | None = None,
        *,
        rows: EllipsisType | tuple[int, int] = ...,
        cols: EllipsisType | tuple[int, int] = ...,
    ) -> Any:
        """
        Creates a new Slice instance with the provided ranges and applies it to the relation.

        Parameters
        ----------
        relation : Any, optional
            The relation to apply the slice to. Default: None.
        rows : EllipsisType | tuple[int, int], optional
            The row range to slice. Default: Ellipsis (all rows).
        cols : EllipsisType | tuple[int, int], optional
            The column range to slice. Default: Ellipsis (all columns).

        Returns
        -------
        TransformationFunction
            The new Slice instance (attached to the relation if provided).
        """
        slice = Slice(self.name, rows=rows, cols=cols)
        return TransformationFunction.__call__(slice, relation)

    def is_parametrized(self) -> bool:
        return True

    def get(self) -> Any:
        cols = None if self.cols is Ellipsis else self.cols
        rows = None if self.rows is Ellipsis else self.rows

        return jpype.JClass("cz.cvut.fel.ida.algebra.functions.transformation.joint.Slice")(rows, cols)

    def wrap(self, content: str) -> str:
        rows = "..." if self.rows is Ellipsis or self.rows is None else self.rows
        cols = "..." if self.cols is Ellipsis or self.cols is None else self.cols

        return f"slice({content}, rows={rows}, cols={cols})"

    def __str__(self) -> str:
        rows = "..." if self.rows is Ellipsis or self.rows is None else self.rows
        cols = "..." if self.cols is Ellipsis or self.cols is None else self.cols

        return f"slice(rows={rows}, cols={cols})"
