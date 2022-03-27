from typing import Optional, Iterable, Sequence

import numpy as np
import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.settings import SettingsProxy, Settings


class JavaFactory:
    def __init__(self, settings: Optional[SettingsProxy] = None):
        from neuralogic.core.constructs.rule import Rule
        from neuralogic.core.constructs.atom import WeightedAtom

        if not is_initialized():
            initialize()

        if settings is None:
            settings = Settings().create_proxy()
        self.settings = settings

        self.weighted_atom_type = WeightedAtom
        self.rule_type = Rule

        self.predicate_metadata = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.metadata.PredicateMetadata")
        self.rule_metadata = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.metadata.RuleMetadata")

        self.template_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.TemplateBuilder")
        self.examples_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.ExamplesBuilder")

        self.var_factory_class = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.factories.VariableFactory")

        self.body_atom = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.components.BodyAtom")
        self.head_atom = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.components.HeadAtom")
        self.weighted_rule = jpype.JClass("cz.cvut.fel.ida.logic.constructs.template.components.WeightedRule")

        self.builder = self.template_builder(settings.settings)

        self.constant_factory = self.builder.constantFactory
        self.predicate_factory = self.builder.predicateFactory
        self.weight_factory = self.builder.weightFactory

        self.lifted_example = jpype.JClass("cz.cvut.fel.ida.logic.constructs.example.LiftedExample")
        self.valued_fact = jpype.JClass("cz.cvut.fel.ida.logic.constructs.example.ValuedFact")

        self.clause = jpype.JClass("cz.cvut.fel.ida.logic.Clause")
        self.literal = jpype.JClass("cz.cvut.fel.ida.logic.Literal")
        self.conjunction = jpype.JClass("cz.cvut.fel.ida.logic.constructs.Conjunction")

        self.string_value = jpype.JClass("cz.cvut.fel.ida.algebra.values.StringValue")
        self.scalar_value = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")
        self.vector_value = jpype.JClass("cz.cvut.fel.ida.algebra.values.VectorValue")
        self.matrix_value = jpype.JClass("cz.cvut.fel.ida.algebra.values.MatrixValue")

        self.pair = jpype.JClass("cz.cvut.fel.ida.utils.generic.Pair")

        self.unit_weight = jpype.JClass("cz.cvut.fel.ida.algebra.weights.Weight").unitWeight
        self.variable_factory = self.get_variable_factory()

    def get_variable_factory(self):
        return self.var_factory_class()

    def get_term(self, term, variable_factory):
        if isinstance(term, str):
            if term[0].islower() or term.isnumeric():
                return self.constant_factory.construct(term)
            elif term[0].isupper():
                return variable_factory.construct(term)
            else:
                raise NotImplementedError
        if isinstance(term, (int, float)):
            return self.constant_factory.construct(str(term))
        raise NotImplementedError

    def atom_to_clause(self, atom):
        terms = [self.get_term(term, self.variable_factory) for term in atom.terms]

        predicate_name = f"@{atom.predicate.name}" if atom.predicate.special else atom.predicate.name
        literal = self.literal(predicate_name, atom.negated, terms)
        literal_array = self.literal[1]
        literal_array[0] = literal

        return self.clause(literal_array)

    def get_generic_atom(self, atom_class, atom, variable_factory, default_weight=None, is_example=False):
        predicate = self.get_predicate(atom.predicate)

        weight = None
        if isinstance(atom, self.weighted_atom_type):
            weight = self.get_weight(atom.weight, atom.weight_name, atom.is_fixed or is_example)
        elif default_weight is not None:
            weight = self.get_weight(default_weight, None, True)

        term_list = [self.get_term(term, variable_factory) for term in atom.terms]

        j_term_list = jpype.java.util.ArrayList()
        for x in term_list:
            j_term_list.add(x)

        java_atom = atom_class(predicate, j_term_list, atom.negated, weight)
        java_atom.originalString = atom.to_str()

        return java_atom

    def get_metadata(self, metadata, metadata_class):
        if metadata is None:
            return None

        if (
            metadata.aggregation is None
            and metadata.activation is None
            and metadata.offset is None
            and metadata.learnable is None
        ):
            return None

        map = jpype.JClass("java.util.LinkedHashMap")()

        if metadata.aggregation is not None:
            map.put("aggregation", self.string_value(metadata.aggregation.value.lower()))
        if metadata.activation is not None:
            map.put("activation", self.string_value(metadata.activation.lower()))
        # if metadata.offset is not None:
        #     _, value = self.get_value(metadata.offset)
        #     map.put("offset", self.weight_factory.construct(value))
        if metadata.learnable is not None:
            map.put("learnable", self.string_value(str(metadata.learnable).lower()))

        return metadata_class(self.builder.settings, map)

    def get_query(self, query):
        variable_factory = self.get_variable_factory()

        if not isinstance(query, self.rule_type):
            if not isinstance(query, Iterable):
                query = [query]
            return None, self.get_conjunction(query, variable_factory, 1.0, True)
        return self.get_atom(query.head, variable_factory, True), self.get_conjunction(
            query.body, variable_factory, is_example=True
        )

    def get_lifted_example(self, example):
        conjunctions = []
        rules = []
        label_conjunction = None

        variable_factory = self.get_variable_factory()

        if not isinstance(example, self.rule_type):
            if not isinstance(example, Iterable):
                example = [example]
            conjunctions.append(self.get_conjunction(example, variable_factory, is_example=True))
        else:
            label_conjunction = self.get_conjunction([example.head], variable_factory, is_example=True)
            conjunctions.append(self.get_conjunction(example.body, variable_factory, is_example=True))

        lifted_example = self.lifted_example(jpype.java.util.ArrayList(conjunctions), jpype.java.util.ArrayList(rules))
        return label_conjunction, lifted_example

    def get_conjunction(self, atoms, variable_factory, default_weight=None, is_example=False):
        valued_facts = [self.get_valued_fact(atom, variable_factory, default_weight, is_example) for atom in atoms]
        return self.conjunction(jpype.java.util.ArrayList(valued_facts))

    def get_predicate_metadata_pair(self, predicate_metadata):
        return self.pair(
            self.get_predicate(predicate_metadata.predicate),
            self.get_metadata(predicate_metadata.metadata, self.predicate_metadata),
        )

    def get_valued_fact(self, atom, variable_factory, default_weight=None, is_example=False):
        return self.get_generic_atom(
            self.valued_fact,
            atom,
            variable_factory,
            default_weight,
            is_example,
        )

    def get_atom(self, atom, variable_factory, is_example=False):
        return self.get_generic_atom(self.body_atom, atom, variable_factory, is_example=is_example)

    def get_rule(self, rule):
        java_rule = self.weighted_rule()
        java_rule.setOriginalString(str(rule))

        variable_factory = self.get_variable_factory()

        head_atom = self.get_atom(rule.head, variable_factory)
        weight = head_atom.getConjunctWeight()

        if weight is None:
            java_rule.setWeight(self.unit_weight)
        else:
            java_rule.setWeight(weight)

        body_atoms = [self.get_atom(atom, variable_factory) for atom in rule.body]
        body_atom_list = jpype.java.util.ArrayList(body_atoms)

        java_rule.setHead(self.head_atom(head_atom))
        java_rule.setBody(body_atom_list)

        offset = None  # TODO: Implement

        java_rule.setOffset(offset)

        if rule.metadata is not None:
            java_rule.allowDuplicitGroundings = rule.metadata.duplicit_grounding

        java_rule.setMetadata(self.get_metadata(rule.metadata, self.rule_metadata))

        return java_rule

    def get_predicate(self, predicate):
        return self.predicate_factory.construct(predicate.name, predicate.arity, predicate.special, predicate.hidden)

    def get_weight(self, weight, name, fixed):
        initialized, value = self.get_value(weight)

        if name is None:
            return self.weight_factory.construct(value, fixed, initialized)
        return self.weight_factory.construct(name, value, fixed, initialized)

    def get_value(self, weight):
        if isinstance(weight, (int, float, np.number)):
            value = self.scalar_value(float(weight))
            initialized = True
        elif isinstance(weight, tuple):
            if len(weight) == 1:
                if weight[0] == 1:
                    value = self.scalar_value()
                else:
                    value = self.vector_value(weight[0])
            elif len(weight) == 2:
                if weight[0] == 1:
                    value = self.vector_value(weight[1])
                    value.rowOrientation = True
                elif weight[1] == 1:
                    value = self.vector_value(weight[0])
                    value.rowOrientation = False
                else:
                    value = self.matrix_value(weight[0], weight[1])
            else:
                raise NotImplementedError
            initialized = False
        elif isinstance(weight, Sequence):
            initialized = True
            if len(weight) == 0:
                raise NotImplementedError
            if isinstance(weight[0], (int, float, np.number)):
                vector = [float(w) for w in weight]
                value = self.vector_value(vector)
            elif isinstance(weight[0], (Sequence, np.ndarray)):
                if len(weight) == 1:
                    value = self.vector_value([float(w) for w in weight[0]])
                    value.rowOrientation = True
                else:
                    try:
                        matrix = [[float(w) for w in weights] for weights in weight]
                        value = self.matrix_value(matrix)
                    except TypeError:
                        value = self.vector_value([float(w) for w in weight])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return initialized, value

    def get_new_weight_factory(self):
        return self.examples_builder(self.settings.settings).weightFactory
