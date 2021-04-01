from py4j.java_gateway import get_field, set_field
from typing import Optional
from py4j.java_collections import ListConverter

from neuralogic import get_neuralogic, get_gateway
from neuralogic.model import factories
from neuralogic.settings import Settings


class JavaFactory:
    def __init__(self, settings: Settings):
        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.building

        self.namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.template.components
        self.value_namespace = get_neuralogic().cz.cvut.fel.ida.algebra.values

        self.builder = namespace.TemplateBuilder(settings.settings)

        self.constant_factory = get_field(self.builder, "constantFactory")
        self.predicate_factory = get_field(self.builder, "predicateFactory")
        self.weight_factory = get_field(self.builder, "weightFactory")

        self.unit_weight = get_neuralogic().cz.cvut.fel.ida.algebra.weights.Weight.unitWeight

    def get_variable_factory(self):
        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.building.factories
        variable_factory = namespace.VariableFactory()

        return variable_factory

    def get_term(self, term, variable_factory):
        if isinstance(term, str):
            if term[0].islower():
                return self.constant_factory.construct(term)
            elif term[0].isupper():
                return variable_factory.construct(term)
            else:
                raise NotImplementedError
        if isinstance(term, (int, float)):
            return self.constant_factory.construct(str(term))
        raise NotImplementedError

    def get_atom(self, atom, variable_factory):
        predicate = self.get_predicate(atom.predicate)
        weight = self.get_weight(atom.weight, atom.is_fixed) if isinstance(atom, factories.WeightedAtom) else None
        term_list = ListConverter().convert(
            [self.get_term(term, variable_factory) for term in atom.terms], get_gateway()._gateway_client
        )
        body_atom = self.namespace.BodyAtom(predicate, term_list, atom.negated, weight)

        return body_atom

    def get_rule(self, rule):
        java_rule = self.namespace.WeightedRule()
        variable_factory = self.get_variable_factory()

        head_atom = self.get_atom(rule.head, variable_factory)
        weight = head_atom.getConjunctWeight()

        if weight is None:
            java_rule.setWeight(self.unit_weight)
        else:
            java_rule.setWeight(weight)

        body_atoms = [self.get_atom(atom, variable_factory) for atom in rule.body]
        body_atom_list = ListConverter().convert(body_atoms, get_gateway()._gateway_client)

        java_rule.setHead(self.namespace.HeadAtom(head_atom))
        java_rule.setBody(body_atom_list)

        offset = None  # TODO: Implement

        java_rule.setOffset(offset)
        java_rule.setMetadata(None)  # TODO: Implement

        return java_rule

    def get_predicate(self, predicate):
        return self.predicate_factory.construct(predicate.name, predicate.arity, predicate.special, predicate.private)

    def get_weight(self, weight, fixed):
        initialized, value = self.get_value(weight)
        return self.weight_factory.construct(value, fixed, initialized)

    def get_value(self, weight):
        if isinstance(weight, (int, float)):
            value = self.value_namespace.ScalarValue(float(weight))
            initialized = True
        elif isinstance(weight, tuple):
            if len(weight) == 1:
                if weight[0] == 1:
                    value = self.value_namespace.ScalarValue()
                else:
                    value = self.value_namespace.VectorValue(weight[0])
            elif len(weight) == 2:
                if weight[0] == 1:
                    value = self.value_namespace.VectorValue(weight[1])
                    set_field(value, "rowOrientation", True)
                elif weight[1] == 1:
                    value = self.value_namespace.VectorValue(weight[0])
                    set_field(value, "rowOrientation", False)
                else:
                    value = self.value_namespace.MatrixValue(weight[0], weight[1])
            else:
                raise NotImplementedError
            initialized = False
        else:
            # TODO: Implement matrix/vector
            raise NotImplementedError
        return initialized, value


java_factory: Optional[JavaFactory] = None


def get_java_factory() -> JavaFactory:
    if java_factory is None:
        raise Exception
    return java_factory


def init_java_factory(settings: Settings) -> JavaFactory:
    global java_factory
    java_factory = JavaFactory(settings)

    return java_factory
