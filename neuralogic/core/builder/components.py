import enum
from typing import Any

import numpy as np


class NeuronType(str, enum.Enum):
    """
    Enum representing different types of neurons in the neural network.
    """

    Aggregation = "AggregationNeuron"
    Atom = "AtomNeuron"
    Negation = "NegationNeuron"
    Rule = "RuleNeuron"
    SplittableAggregation = "SplittableAggregationNeuron"
    WeightedAtom = "WeightedAtomNeuron"
    WeightedRule = "WeightedRuleNeuron"
    Fact = "FactNeuron"


class Atom:
    """
    Represents an atom in the logic program, often corresponding to a node in the neural network.
    """

    __slots__ = "substitutions", "_atom", "_predicate", "_arity"

    def __init__(self, atom: Any, substitutions: dict[str, Any]):
        """
        Parameters
        ----------
        atom : Any
            The underlying Java atom object.
        substitutions : Dict
            Dictionary of variable substitutions.
        """
        self.substitutions = substitutions
        self._atom = atom

        self._predicate = atom.predicateName()
        self._arity = atom.arity()

    @property
    def predicate(self) -> str:
        return self._predicate

    @property
    def arity(self) -> int:
        return self._arity

    def node_type(self) -> NeuronType:
        """
        Returns the type of the neuron.

        Returns
        -------
        NeuronType
            The type of the neuron.
        """
        return NeuronType(self._atom.getClass().getSimpleName())

    def __str__(self) -> str:
        return str(self._atom)


class Neuron(Atom):
    """
    Represents a neuron in the neural network, extending the Atom class with value and gradient properties.
    """

    def __init__(self, neuron: Any, substitutions: dict[str, Any]):
        self.substitutions = substitutions
        self._atom = neuron

        self._predicate = neuron.getName()
        self._arity = len(substitutions)

    @property
    def value(self) -> float | list | np.ndarray:
        from neuralogic.core.constructs.java_objects import ValueFactory

        return ValueFactory.from_java(self._atom.getRawState().getValue())

    @property
    def gradient(self) -> float | list | np.ndarray:
        from neuralogic.core.constructs.java_objects import ValueFactory

        return ValueFactory.from_java(self._atom.getRawState().getGradient())


class NeuralSample:
    """
    Represents a single training or testing sample, containing the query and its associated neural network (evidence).
    """

    __slots__ = "_java_sample", "_neurons"

    def __init__(self, sample: Any):
        self._java_sample = sample
        self._neurons = None

    @property
    def neurons(self) -> dict[str, dict[str, dict[tuple[str, ...], Any]]]:
        if self._neurons is None:
            self._neurons = self._get_neurons()
        return self._neurons

    @property
    def target(self) -> float | list | np.ndarray:
        from neuralogic.core.constructs.java_objects import ValueFactory

        return ValueFactory.from_java(self._java_sample.target)

    def get_neurons(self, literal: Any, neuron_type: NeuronType | None = NeuronType.Atom) -> list[Neuron]:
        """
        Returns a list of neurons matching the provided literal and neuron type.

        Parameters
        ----------
        literal : Any
            The literal to match.
        neuron_type : NeuronType, optional
            The type of neurons to search for. Default: NeuronType.Atom.

        Returns
        -------
        list[Neuron]
            The list of matching neurons.
        """
        literal_name = literal.predicate.name
        literal_arity = literal.predicate.arity

        neurons = []
        neurons_by_name_vals = []

        if neuron_type is None:
            neurons_by_name_vals = self.neurons.values()
        elif neuron_type in self.neurons:
            neurons_by_name_vals = [self.neurons[neuron_type]]

        for nodes_by_name in neurons_by_name_vals:
            if literal_name not in nodes_by_name:
                continue

            for subs, value in nodes_by_name[literal_name].items():
                if len(subs) != literal_arity:
                    continue

                literal_subs = {}
                for term, sub in zip(literal.terms, subs):
                    term_str = str(term)

                    if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                        if term_str in literal_subs and sub != literal_subs[term_str]:
                            break
                        literal_subs[str(term)] = sub
                        continue

                    if str(term) != sub:
                        break
                else:
                    neurons.append(Neuron(value, literal_subs))
        return neurons

    def _get_neurons(self) -> dict[str, dict[str, dict[tuple[str, ...], Any]]]:
        nodes = {}

        for node in self._java_sample.query.evidence.allNeuronsTopologic:
            node_type = str(node.getClass().getSimpleName())
            name = str(node.name).strip()

            bracket = name.rfind("(")
            space = name.rfind(" ", 0, bracket if bracket != -1 else None)

            substitutions = tuple()
            if bracket != -1:
                subs = name[bracket + 1 :]
                name = name[space + 1 : bracket]

                r_bracket = subs.find(")")
                substitutions = tuple(subs[:r_bracket].split(", "))
            elif space != -1:
                name = name[space + 1 :]

            if node_type not in nodes:
                nodes[node_type] = {}

            if name not in nodes[node_type]:
                nodes[node_type][name] = {}

            nodes[node_type][name][substitutions] = node
        return nodes

    def get_fact(self, fact: Any) -> list[Neuron]:
        """
        Returns the neuron corresponding to the provided fact.

        Parameters
        ----------
        fact : Any
            The fact to look for.

        Returns
        -------
        list[Neuron]
            The matching fact neuron(s).
        """
        for term in fact.terms:
            term_str = str(term)

            if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                raise ValueError(f"{fact} is not a fact")

        return self.get_neurons(fact, NeuronType.Fact)

    def set_fact_value(self, fact: Any, value: float) -> int:
        """
        Sets the value of a specific fact in the sample.

        Parameters
        ----------
        fact : Any
            The fact to set the value for.
        value : float
            The value to set.

        Returns
        -------
        int
            The index of the fact neuron, or -1 if not found.
        """
        for term in fact.terms:
            term_str = str(term)

            if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                raise ValueError(f"{fact} is not a fact")

        node = self.get_neurons(fact, NeuronType.Fact)

        if len(node) == 0:
            return -1

        sample_fact = node[0]._atom
        sample_fact.getRawState().setValue(value)
        sample_fact.offset.value = value

        return sample_fact.index

    def draw(
        self,
        filename: str | None = None,
        show: bool = True,
        img_type: str = "png",
        value_detail: int = 0,
        graphviz_path: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Draws the neural sample.

        Parameters
        ----------
        filename : str, optional
            The filename to save the drawing to. Default: None.
        show : bool
            Whether to show the drawing. Default: True.
        img_type : str
            The image type. Default: "png".
        value_detail : int
            The level of detail for values. Default: 0.
        graphviz_path : str, optional
            The path to the Graphviz executable. Default: None.

        Returns
        -------
        Any
            The drawing data or image object.
        """
        from neuralogic.utils.visualize import draw_sample

        return draw_sample(self, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __str__(self) -> str:
        return str(self._java_sample)


class Grounding:
    """
    Represents a grounded model, providing access to grounded atoms and facts.
    """

    __slots__ = ("_grounding", "_atoms")

    def __init__(self, grounding: Any):
        self._grounding = grounding
        self._atoms = None

    def draw(
        self,
        filename: str | None = None,
        show: bool = True,
        img_type: str = "png",
        value_detail: int = 0,
        graphviz_path: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Draws the grounding.

        Parameters
        ----------
        filename : str, optional
            The filename to save the drawing to. Default: None.
        show : bool
            Whether to show the drawing. Default: True.
        img_type : str
            The image type. Default: "png".
        value_detail : int
            The level of detail for values. Default: 0.
        graphviz_path : str, optional
            The path to the Graphviz executable. Default: None.

        Returns
        -------
        Any
            The drawing data or image object.
        """
        from neuralogic.utils.visualize import draw_grounding

        return draw_grounding(self._grounding, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __hash__(self):
        return hash(self._grounding)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grounding):
            return False
        return other._grounding == self._grounding

    def __str__(self) -> str:
        return str(self._grounding)

    @property
    def atoms(self) -> dict[str, dict[tuple[str, ...], Any]]:
        if self._atoms is None:
            self._atoms = self._get_atoms()
        return self._atoms

    def get_atoms(self, literal: Any) -> list[Atom]:
        """
        Returns a list of grounded atoms matching the provided literal.

        Parameters
        ----------
        literal : Any
            The literal to match.

        Returns
        -------
        list[Atom]
            The list of matching grounded atoms.
        """
        literal_name = literal.predicate.name
        literal_arity = literal.predicate.arity

        if literal_name not in self.atoms:
            return []

        nodes = []
        for subs, value in self.atoms[literal_name].items():
            if len(subs) != literal_arity:
                continue

            literal_subs = {}
            for term, sub in zip(literal.terms, subs):
                term_str = str(term)

                if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                    if term_str in literal_subs and sub != literal_subs[term_str]:
                        break
                    literal_subs[str(term)] = sub
                    continue

                if str(term) != sub:
                    break
            else:
                nodes.append(Atom(value, literal_subs))
        return nodes

    def _get_atoms(self) -> dict[str, dict[tuple[str, ...], Any]]:
        atoms = {}

        for literal in self._grounding.groundingWrap.getGroundTemplate().derivedGroundFacts:
            self._process_literal(literal, atoms)

        for literal in self._grounding.groundingWrap.getGroundTemplate().groundFacts:
            self._process_literal(literal, atoms)

        for literal in self._grounding.groundingWrap.getGroundTemplate().templateFacts:
            self._process_literal(literal, atoms)

        return atoms

    def _process_literal(self, literal: Any, atoms: dict[str, dict[tuple[str, ...], Any]]) -> None:
        name = str(literal).strip()

        bracket = name.rfind("(")
        space = name.rfind(" ", 0, bracket if bracket != -1 else None)

        substitutions = tuple()
        if bracket != -1:
            subs = name[bracket + 1 :]
            name = name[space + 1 : bracket]

            r_bracket = subs.find(")")
            substitutions = tuple(subs[:r_bracket].split(", "))
        elif space != -1:
            name = name[space + 1 :]

        if name not in atoms:
            atoms[name] = {}

        atoms[name][substitutions] = literal
