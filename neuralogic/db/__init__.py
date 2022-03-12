from collections import defaultdict
from typing import Dict, Tuple, List

from neuralogic.core import Settings, Activation, Aggregation
from neuralogic.core.constructs.atom import WeightedAtom, BaseAtom
from neuralogic.core.constructs.predicate import PredicateMetadata, Predicate
from neuralogic.core.constructs.rule import Rule

from neuralogic.db.sql_convertors import (
    _get_fact_sql_function,
    _get_rule_sql_function,
    _get_rule_aggregation_sql_function,
    _get_relation_interface_sql_function,
)
from neuralogic.db.sql_helpers import helpers


def to_sql(model, mapping: Dict[Predicate, Tuple[str, Tuple[str], str]], settings: Settings) -> str:
    """Converts the model into SQL"""
    template = model.source_template
    batched_relations = defaultdict(lambda: defaultdict(list))
    predicates_metadata = {}
    mapping = {f"{key.name}/{key.arity}": value for key, value in mapping.items()}

    weight_index = 0
    weights = model.state_dict()["weights"]

    activations = {
        str(Activation.SIGMOID).lower(): "pynelo_sigmoid",
        str(Activation.TANH).lower(): "pynelo_tanh",
        str(Activation.IDENTITY).lower(): "",
    }

    aggregations = {
        str(Aggregation.AVG).lower(): "AVG",
    }

    rule_default_activation = str(settings.rule_neuron_activation).lower()
    relation_default_activation = str(settings.relation_neuron_activation).lower()
    default_aggregation = str(Aggregation.AVG).lower()

    for rule in template:
        if isinstance(rule, Rule):
            weight_indices = []
            if isinstance(rule.head, WeightedAtom) and rule.head.weight is not None:
                weight_indices.append(weight_index)
                weight_index += 1
            else:
                weight_indices.append(None)
            for body_relation in rule.body:
                if isinstance(body_relation, WeightedAtom) and body_relation.weight is not None:
                    weight_indices.append(weight_index)
                    weight_index += 1
                else:
                    weight_indices.append(None)
            batched_relations[rule.head.predicate.name][rule.head.predicate.arity].append((rule, weight_indices))
        elif isinstance(rule, (WeightedAtom, BaseAtom)):
            if isinstance(rule, WeightedAtom) and rule.weight is not None:
                weight_indices = [weight_index]
                weight_index += 1
            else:
                weight_indices = [None]
            batched_relations[rule.predicate.name][rule.predicate.arity].append((rule, weight_indices))
        elif isinstance(rule, PredicateMetadata):
            predicates_metadata[str(rule.predicate)] = rule.metadata
        else:
            raise NotImplementedError("Template can contain only relations or predicate metadata!")

    sql_source = [helpers]

    for name, arities in batched_relations.items():
        for arity, relations_by_arity in arities.items():
            if f"{name}/{arity}" in mapping:
                raise Exception

            activation = relation_default_activation
            aggregation = default_aggregation
            predicate_metadata = predicates_metadata.get(f"{name}/{arity}", None)

            if predicate_metadata is not None:
                if predicate_metadata.activation is not None:
                    activation = str(predicate_metadata.activation).lower()
                if predicate_metadata.aggregation is not None:
                    aggregation = str(predicate_metadata.aggregation).lower()

            for index, (relation, weight_indices) in enumerate(relations_by_arity):
                if isinstance(relation, Rule):
                    act, agg = rule_default_activation, default_aggregation

                    if relation.metadata is not None:
                        if relation.metadata.activation is not None:
                            act = str(relation.metadata.activation).lower()
                        if relation.metadata.aggregation is not None:
                            agg = str(relation.metadata.aggregation).lower()

                    sql_fun = _get_rule_sql_function(
                        relation, index, activations[act], aggregations[agg], weight_indices, weights, mapping
                    )
                else:
                    sql_fun = _get_fact_sql_function(relation, index, weight_indices, weights)
                sql_source.append(sql_fun)
            sql_source.append(
                _get_rule_aggregation_sql_function(
                    name,
                    arity,
                    len(relations_by_arity),
                    activations[activation],
                    aggregations[aggregation],
                )
            )
            sql_source.append(_get_relation_interface_sql_function(name, arity))
    return "\n".join(sql_source)
