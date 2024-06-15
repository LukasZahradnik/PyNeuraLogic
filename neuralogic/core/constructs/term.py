from typing import Optional, Union, List


class Variable:
    __slots__ = "name", "type"

    def __init__(self, name: str, type: Optional[str] = None):
        self.name = name
        self.type = type

    def __str__(self) -> str:
        if self.type is not None:
            return f"{self.type}:{self.name}"
        return f"{self.name}"

    def __getitem__(self, item) -> List["Variable"]:
        if not isinstance(item, slice):
            raise ValueError("Variable range can be only defined by a slice")

        if item.step is not None:
            return [Variable(f"{self.name}{i}", self.type) for i in range(item.start, item.stop, item.step)]

        return [Variable(f"{self.name}{i}", self.type) for i in range(item.start, item.stop)]

    def __call__(self, item: str) -> "Variable":
        if isinstance(item, str):
            return Variable(self.name, item)
        raise ValueError("Type can be only of type str")

    def __eq__(self, other: Union["Variable", str]) -> bool:
        return str(other) == str(self)

    def __hash__(self):
        return str(self).__hash__()

    def __lt__(self, other):
        return str(self).__lt__(str(other))


class Constant:
    __slots__ = "name", "type"

    def __init__(self, name: str, type: Optional[str] = None):
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

    def __eq__(self, other: Union["Constant", str]) -> bool:
        return str(other) == str(self)

    def __hash__(self):
        return str(self).__hash__()

    def __lt__(self, other):
        return str(self).__lt__(str(other))
