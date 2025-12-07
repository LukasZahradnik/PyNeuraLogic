import enum
from typing import Any, Dict, Optional

from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.utils.visualize import draw_sample, draw_grounding


class NeuronType(enum.StrEnum):
    Aggregation = "AggregationNeuron"
    Atom = "AtomNeuron"
    Negation = "NegationNeuron"
    Rule = "RuleNeuron"
    SplittableAggregation = "SplittableAggregationNeuron"
    WeightedAtom = "WeightedAtomNeuron"
    WeightedRule = "WeightedRuleNeuron"
    Fact = "FactNeuron"


class Atom:
    __slots__ = "substitutions", "_atom", "_predicate", "_arity"

    def __init__(self, atom, substitutions: Dict):
        self.substitutions = substitutions
        self._atom = atom

        self._predicate = atom.predicateName()
        self._arity = atom.arity()

    @property
    def predicate(self):
        return self._predicate

    @property
    def arity(self):
        return self._arity

    def node_type(self) -> NeuronType:
        return NeuronType(self._atom.getClass().getSimpleName())

    def __str__(self):
        return str(self._atom)


class Neuron(Atom):
    def __init__(self, neuron, substitutions: Dict):
        self.substitutions = substitutions
        self._atom = neuron

        self._predicate = neuron.getName()
        self._arity = len(substitutions)

    @property
    def value(self):
        return ValueFactory.from_java(self._atom.getRawState().getValue())

    @property
    def gradient(self):
        return ValueFactory.from_java(self._atom.getRawState().getGradient())


class NeuralSample:
    __slots__ = "_java_sample", "_neurons"

    def __init__(self, sample):
        self._java_sample = sample
        self._neurons = None

    @property
    def neurons(self):
        if self._neurons is None:
            self._neurons = self._get_neurons()
        return self._neurons

    @property
    def target(self):
        return ValueFactory.from_java(self._java_sample.target)

    def get_neurons(self, literal, neuron_type: NeuronType | None = NeuronType.Atom):
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

    def _get_neurons(self):
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

    def get_fact(self, fact):
        for term in fact.terms:
            term_str = str(term)

            if term_str[0] == term_str[0].upper() and term_str[0] != term_str[0].lower():
                raise ValueError(f"{fact} is not a fact")

        return self.get_neurons(fact, NeuronType.Fact)

    def set_fact_value(self, fact, value) -> int:
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
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: str | None = None,
        *args,
        **kwargs,
    ):
        return draw_sample(self, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __str__(self):
        return str(self._java_sample)


class Grounding:
    __slots__ = ("_grounding", "_atoms")

    def __init__(self, grounding):
        self._grounding = grounding
        self._atoms = None

    def draw(
        self,
        filename: Optional[str] = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return draw_grounding(self._grounding, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def __hash__(self):
        return hash(self._grounding)

    def __eq__(self, other):
        if not isinstance(other, Grounding):
            return False
        return other._grounding == self._grounding

    def __str__(self):
        return str(self._grounding)

    @property
    def atoms(self) -> dict[str, dict[tuple[str, ...], Any]]:
        if self._atoms is None:
            self._atoms = self._get_atoms()
        return self._atoms

    def get_atoms(self, literal) -> list[Atom]:
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

    def _get_atoms(self):
        atoms = {}

        for literal in self._grounding.groundingWrap.getGroundTemplate().derivedGroundFacts:
            self._process_literal(literal, atoms)

        for literal in self._grounding.groundingWrap.getGroundTemplate().groundFacts:
            self._process_literal(literal, atoms)

        return atoms

    def _process_literal(self, literal, atoms):
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
