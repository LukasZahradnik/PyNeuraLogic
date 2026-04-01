from typing import Any


class Variable:
    """
    Represents a variable in a logic expression.
    """

    __slots__ = "name", "type"

    def __init__(self, name: str, type: str | None = None):
        """
        Parameters
        ----------
        name : str
            The name of the variable.
        type : str, optional
            The type of the variable. Default: None.
        """
        self.name = name
        self.type = type

    def __str__(self) -> str:
        if self.type is not None:
            return f"{self.type}:{self.name}"
        return f"{self.name}"

    def __getitem__(self, item: slice) -> list["Variable"]:
        if not isinstance(item, slice):
            raise ValueError("Variable range can be only defined by a slice")

        if item.step is not None:
            return [Variable(f"{self.name}{i}", self.type) for i in range(item.start, item.stop, item.step)]

        return [Variable(f"{self.name}{i}", self.type) for i in range(item.start, item.stop)]

    def __call__(self, item: str) -> "Variable":
        if isinstance(item, str):
            return Variable(self.name, item)
        raise ValueError("Type can be only of type str")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Variable, str)):
            raise NotImplementedError
        return str(other) == str(self)

    def __hash__(self) -> int:
        return str(self).__hash__()

    def __lt__(self, other: Any) -> bool:
        return str(self).__lt__(str(other))


class Constant:
    """
    Represents a constant in a logic expression.
    """

    __slots__ = "name", "type"

    def __init__(self, name: str, type: str | None = None):
        """
        Parameters
        ----------
        name : str
            The name of the constant.
        type : str, optional
            The type of the constant. Default: None.
        """
        self.name = name
        self.type = type

    def __str__(self) -> str:
        if self.type is not None:
            return f"{self.type}:{self.name}"
        return f"{self.name}"

    def __call__(self, item: str) -> "Constant":
        if isinstance(item, str):
            return Constant(self.name, item)
        raise ValueError("Type can be only of type str")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Constant, str)):
            raise NotImplementedError
        return str(other) == str(self)

    def __hash__(self) -> int:
        return str(self).__hash__()

    def __lt__(self, other: Any) -> bool:
        return str(self).__lt__(str(other))
