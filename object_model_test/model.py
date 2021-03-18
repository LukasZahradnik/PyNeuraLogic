from typing import Iterable, Generator


class AtomBase:
    def __init__(self, name):
        self.name = name
        self.children = None
        self.can_have_child = True

    def __call__(self, *args) -> "RuleAtom":
        return RuleAtom(self.name, args)

    def __getitem__(self, item) -> "RuleAtom":
        return RuleAtom(self.name, None)[item]

    def __gt__(self, other) -> "RuleAtom":
        atom = RuleAtom(self.name)
        atom.can_have_child = self.can_have_child
        return atom > other

    def __str__(self) -> str:
        if self.children is None:
            return f"{self.name}."
        return self.name


class RuleAtom(AtomBase):
    def __init__(self, name, args=None):
        super().__init__(name)

        self.args = args
        self.weights = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item) -> "RuleAtom":
        self.weights = item
        return self

    def __gt__(self, other) -> "RuleAtom":
        if not self.can_have_child:
            raise NotImplementedError

        if isinstance(other, (RuleAtom, AtomBase)):
            other.can_have_child = False
            other.children = None
        elif isinstance(other, Iterable):
            if isinstance(other, Generator):
                other_store = []

                for atom in other:
                    atom.can_have_child = False
                    atom.children = None
                    other_store.append(atom)
                other = other_store
            else:
                for atom in other:
                    atom.can_have_child = False
                    atom.children = None
        self.children = other
        return self

    def __str__(self) -> str:
        children = ""
        weights = ""
        args = ""

        if self.children is not None:
            if isinstance(self.children, RuleAtom):
                children = f" :- {self.children}"
            elif isinstance(self.children, Iterable):
                children = f" :- {', '.join(str(child) for child in self.children)}"
            else:
                raise Exception

        if self.args is not None:
            if isinstance(self.args, str):
                args = f"({self.args})"
            elif isinstance(self.args, Iterable):
                args = f"({', '.join(str(x) for x in self.args)})"

        if self.weights is not None:
            if isinstance(self.weights, (str, int)):
                weights = f"{{{self.weights}}} "
            elif isinstance(self.weights, Iterable):
                weights = f"{{{', '.join(str(x) for x in self.weights)}}} "

        if self.can_have_child:
            return f"{weights}{self.name}{args}{children}."
        return f"{weights}{self.name}{args}{children}"


class AtomFactory:
    def __getattr__(self, item) -> AtomBase:
        return AtomBase(item)


class VariableFactory:
    def __getattr__(self, item) -> str:
        return item.upper()


Var = VariableFactory()
Atom = AtomFactory()
