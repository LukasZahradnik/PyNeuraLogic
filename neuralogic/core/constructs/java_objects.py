import json
from typing import Optional, Iterable, Sequence

import numpy as np
import jpype

from neuralogic.setup import is_initialized, initialize
from neuralogic.core.constructs.factories import R
from neuralogic.core.constructs.function.enum import Combination
from neuralogic.core.constructs.function.function import CombinationFunction
from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import FContainer
from neuralogic.core.constructs.term import Variable, Constant
from neuralogic.core.settings import SettingsProxy, Settings


class ValueFactory:
    def __init__(self):
        self.scalar_value = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")
        self.vector_value = jpype.JClass("cz.cvut.fel.ida.algebra.values.VectorValue")
        self.matrix_value = jpype.JClass("cz.cvut.fel.ida.algebra.values.MatrixValue")

    @staticmethod
    def from_java(value):
        size = list(value.size())
        if len(size) == 0 or size[0] == 0:
            return float(value.get(0))
        if len(size) == 2 and size[1] != 1:
            return np.array(memoryview(value.getAsArray())).reshape(size).tolist()
        return np.array(memoryview(value.getAsArray())).tolist()

    def get_value(self, weight):
        if isinstance(weight, (float, int)) or np.ndim(weight) == 0:
            return True, self.scalar_value(float(weight))

        if isinstance(weight, tuple):
            if len(weight) == 1:
                if weight[0] == 1:
                    value = self.scalar_value()
                else:
                    value = self.vector_value(weight[0])
            elif len(weight) == 2:
                if weight[0] == 1:
                    value = self.scalar_value() if weight[1] == 1 else self.vector_value(weight[1], True)
                elif weight[1] == 1:
                    value = self.vector_value(weight[0], False)
                else:
                    value = self.matrix_value(weight[0], weight[1])
            else:
                raise NotImplementedError(f"dimensions of size {len(weight)} are not supported. If you wanted to provide tensor as weight, wrap it into a list first")
            return False, value

        if isinstance(weight, (Sequence, np.ndarray, Iterable)):
            if len(weight) == 0:
                raise NotImplementedError

            if isinstance(weight[0], (float, int)) or np.ndim(weight[0]) == 0:
                vector = [float(w) for w in weight]
                return True, self.vector_value(vector)

            if isinstance(weight[0], (Sequence, np.ndarray, Iterable)):
                if len(weight) == 1:
                    value = self.vector_value([float(w) for w in weight[0]])
                    value.rowOrientation = True
                else:
                    try:
                        matrix = [float(w) for weights in weight for w in weights]
                        value = self.matrix_value(matrix, len(weight), len(weight[0]))
                    except TypeError:
                        value = self.vector_value([float(w) for w in weight])
                return True, value

        raise ValueError(f"Cannot create weight from type {type(weight)}, value {weight}")


def _is_body_flat(body: FContainer):
    if not isinstance(body.function, CombinationFunction):
        return False

    for node in body.nodes:
        if isinstance(node, FContainer):
            return False
    return True


def _flatten_rule_body(body, metadata: Metadata):
    combination = Combination.SUM if metadata is None or metadata.combination is None else metadata.combination
    return combination(*body)


class JavaFactory:
    def __init__(self, settings: Optional[SettingsProxy] = None):
        from neuralogic.core.constructs.rule import Rule
        from neuralogic.core.constructs.relation import WeightedRelation

        if not is_initialized():
            initialize()

        if settings is None:
            settings = Settings().create_proxy()

        self.settings = settings

        self.value_factory = ValueFactory()

        self.weighted_atom_type = WeightedRelation
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

        self.transformation = jpype.JClass("cz.cvut.fel.ida.algebra.functions.Transformation")

        self.parameter = jpype.JClass("cz.cvut.fel.ida.algebra.utils.metadata.Parameter")
        self.parameter_val = jpype.JClass("cz.cvut.fel.ida.algebra.utils.metadata.ParameterValue")

        self.pair = jpype.JClass("cz.cvut.fel.ida.utils.generic.Pair")
        self.settings_class = settings.settings_class

        self.unit_weight = jpype.JClass("cz.cvut.fel.ida.algebra.weights.Weight").unitWeight
        self.variable_factory = self.get_variable_factory()

    def get_variable_factory(self):
        return self.var_factory_class()

    def get_term(self, term, variable_factory):
        if isinstance(term, Variable):
            if term.type is None:
                return variable_factory.construct(term.name)
            return variable_factory.construct(term.name, term.type)
        if isinstance(term, Constant):
            if term.type is None:
                return self.constant_factory.construct(term.name)
            return self.constant_factory.construct(term.name, term.type)

        if isinstance(term, str):
            if term[0].islower() or term.isnumeric():
                return self.constant_factory.construct(term)
            elif term[0].isupper():
                return variable_factory.construct(term)
            else:
                raise ValueError(f"Invalid term {term}")
        if isinstance(term, (int, float)):
            return self.constant_factory.construct(str(term))
        raise ValueError(f"Invalid term {term}")

    def atom_to_clause(self, atom):
        terms = [self.get_term(term, self.variable_factory) for term in atom.terms]

        predicate_name = f"@{atom.predicate.name}" if atom.predicate.special else atom.predicate.name
        literal = self.literal(predicate_name, False, terms)
        literal_array = self.literal[1]
        literal_array[0] = literal

        return self.clause(literal_array)

    def to_clause(self, atoms):
        literal_array = self.literal[len(atoms)]

        for i, atom in enumerate(atoms):
            terms = [self.get_term(term, self.variable_factory) for term in atom.terms]
            predicate_name = f"_{atom.predicate.name}" if atom.predicate.hidden else atom.predicate.name

            literal_array[i] = self.literal(predicate_name, atom.negated, terms)
        return self.clause(literal_array)

    def get_generic_relation(self, relation_class, relation, variable_factory, default_weight=None, is_example=False):
        predicate = self.get_predicate(relation.predicate)

        weight = None
        if isinstance(relation, self.weighted_atom_type):
            weight = self.get_weight(relation.weight, relation.weight_name, relation.is_fixed or is_example)
        elif default_weight is not None:
            weight = self.get_weight(default_weight, None, True)

        term_list = [self.get_term(term, variable_factory) for term in relation.terms]

        j_term_list = jpype.java.util.ArrayList()
        for x in term_list:
            j_term_list.add(x)

        if relation_class == self.body_atom:
            if relation.function is None:
                transformation_function = None
            elif relation.function.is_parametrized():
                transformation_function = relation.function.get()
            else:
                function = self.settings_class.parseTransformation(str(relation.function))
                transformation_function = self.transformation.getFunction(function) if function is not None else None

            java_relation = relation_class(predicate, j_term_list, relation.negated, transformation_function, weight)
        else:
            java_relation = relation_class(predicate, j_term_list, relation.negated, weight)
        java_relation.originalString = relation.to_str()

        return java_relation

    def add_metadata_function(self, metadata, map, function: str):
        value = getattr(metadata, function)

        if value is not None and not value.is_parametrized():
            map.put(function, self.string_value(str(value).lower()))

    def add_parametrized_function(self, metadata, metadata_obj, function: str):
        value = getattr(metadata, function)

        if value is not None and value.is_parametrized():
            parameter = self.parameter(function)
            parameter_val = self.parameter_val("dummy")
            parameter_val.value = value.get()
            metadata_obj.put(parameter, parameter_val)

    def get_metadata(self, metadata, metadata_class):
        if metadata is None:
            return None

        if (
            metadata.aggregation is None
            and metadata.transformation is None
            and metadata.combination is None
            and metadata.learnable is None
        ):
            return None

        map = jpype.JClass("java.util.LinkedHashMap")()

        self.add_metadata_function(metadata, map, "aggregation")
        self.add_metadata_function(metadata, map, "transformation")
        self.add_metadata_function(metadata, map, "combination")

        if metadata.learnable is not None:
            map.put("learnable", self.string_value(str(metadata.learnable).lower()))

        metadata_obj = metadata_class(self.builder.settings, map)

        self.add_parametrized_function(metadata, metadata_obj, "aggregation")
        self.add_parametrized_function(metadata, metadata_obj, "transformation")
        self.add_parametrized_function(metadata, metadata_obj, "combination")

        return metadata_obj

    def get_query(self, query):
        variable_factory = self.get_variable_factory()

        if query is None:
            return None, None

        if not isinstance(query, self.rule_type):
            if not isinstance(query, Iterable):
                query = [query]
            if len(query) == 1 and query[0] is None:
                return None, None

            return None, [self.get_valued_fact(relation, variable_factory, 1.0, True) for relation in query]
        return self.get_relation(query.head, variable_factory, True), [
            self.get_valued_fact(relation, variable_factory, True) for relation in query.body
        ]

    def get_lifted_example(self, example, learnable_facts=False):
        conjunctions = []
        rules = []
        label_conjunction = None

        variable_factory = self.get_variable_factory()

        if not isinstance(example, self.rule_type):
            if not isinstance(example, Iterable):
                example = [example]
            conjunctions.append(self.get_conjunction(example, variable_factory, is_example=not learnable_facts))
        else:
            label_conjunction = self.get_conjunction([example.head], variable_factory, is_example=not learnable_facts)
            conjunctions.append(self.get_conjunction(example.body, variable_factory, is_example=not learnable_facts))

        lifted_example = self.lifted_example(jpype.java.util.ArrayList(conjunctions), jpype.java.util.ArrayList(rules))
        return label_conjunction, lifted_example

    def get_conjunction(self, relations, variable_factory, default_weight=None, is_example=False):
        valued_facts = [
            self.get_valued_fact(relation, variable_factory, default_weight, is_example) for relation in relations
        ]
        return self.conjunction(jpype.java.util.ArrayList(valued_facts))

    def get_predicate_metadata_pair(self, predicate_metadata):
        return self.pair(
            self.get_predicate(predicate_metadata.predicate),
            self.get_metadata(predicate_metadata.metadata, self.predicate_metadata),
        )

    def get_valued_fact(self, relation, variable_factory, default_weight=None, is_example=False):
        return self.get_generic_relation(
            self.valued_fact,
            relation,
            variable_factory,
            default_weight,
            is_example,
        )

    def get_relation(self, relation, variable_factory, is_example=False):
        return self.get_generic_relation(self.body_atom, relation, variable_factory, is_example=is_example)

    def get_rule(self, rule):
        java_rule = self.weighted_rule()
        java_rule.setOriginalString(str(rule))

        variable_factory = self.get_variable_factory()

        head_relation = self.get_relation(rule.head, variable_factory)
        weight = head_relation.getConjunctWeight()

        if weight is None:
            java_rule.setWeight(self.unit_weight)
        else:
            java_rule.setWeight(weight)

        body = rule.body
        if not isinstance(body, FContainer) and rule._contains_function_container():
            body = _flatten_rule_body(body, rule.metadata)

        contains_refs = False
        all_variables = {term for term in rule.head.terms if term is not Ellipsis and str(term)[0].isupper()}
        body_relation = []
        all_diff_index = []
        processed_relations = {}
        is_fcontainer = isinstance(body, FContainer)

        for i, relation in enumerate(rule.body):
            all_variables.update(term for term in relation.terms if term is not Ellipsis and str(term)[0].isupper())

            if is_fcontainer:
                if id(relation) in processed_relations:
                    contains_refs = True
                    continue
                processed_relations[id(relation)] = True

            if relation.predicate.special and relation.predicate.name == "alldiff":
                found = False

                for term in relation.terms:
                    if term is Ellipsis:
                        body_relation.append(R.special.alldiff(relation.terms))
                        all_diff_index.append(i)
                        found = True

                        break
                if found:
                    continue

            body_relation.append(self.get_relation(relation, variable_factory))

        for index in all_diff_index:
            terms = {term for term in body_relation[index].terms if term is not Ellipsis}
            terms.update(term for term in all_variables)

            body_relation[index] = self.get_relation(R.special.alldiff(terms), variable_factory)

        body_relation_list = jpype.java.util.ArrayList(body_relation)

        java_rule.setHead(self.head_atom(head_relation))
        java_rule.setBody(body_relation_list)

        offset = None  # TODO: Implement

        java_rule.setOffset(offset)

        if rule.metadata is not None:
            java_rule.allowDuplicitGroundings = bool(rule.metadata.duplicate_grounding)

        metadata = rule.metadata
        if isinstance(body, FContainer):
            metadata = metadata.copy() if metadata is not None else Metadata()

            if not contains_refs and _is_body_flat(body):
                metadata.combination = body.function
                body = body.nodes
            else:
                metadata.combination = body.to_function()

        java_rule.setMetadata(self.get_metadata(metadata, self.rule_metadata))

        return java_rule

    def get_predicate(self, predicate):
        return self.predicate_factory.construct(predicate.name, predicate.arity, predicate.special, predicate.hidden)

    def get_weight(self, weight, name, fixed):
        initialized, value = self.value_factory.get_value(weight)

        if name is None:
            return self.weight_factory.construct(value, fixed, initialized)
        return self.weight_factory.construct(name, value, fixed, initialized)

    def get_new_weight_factory(self):
        return self.examples_builder(self.settings.settings).weightFactory
