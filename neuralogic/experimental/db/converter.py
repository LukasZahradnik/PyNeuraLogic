import dataclasses
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional

from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation

from neuralogic.core import Aggregation, Settings, Metadata
from neuralogic.core.constructs.predicate import PredicateMetadata
from neuralogic.core.constructs.rule import Rule


@dataclasses.dataclass
class TableMapping:
    relation_name: str
    table_name: str
    term_columns: List[str]
    value_column: Optional[str] = None


class Converter:
    def __init__(self, model, table_mappings: List[TableMapping], settings: Settings):
        self.table_mappings: Dict[str, TableMapping] = {
            f"{mapping.relation_name}/{len(mapping.term_columns)}": mapping for mapping in table_mappings
        }

        self.model = model
        self.settings = settings

        self._used_functions: Set[str] = set()

        self.sql_source = None
        self.std_functions = None

    def _process_template_entries(self) -> Tuple[defaultdict, Dict[str, Metadata]]:
        template = self.model.source_template
        weight_index = 0

        batched_relations: defaultdict[str, defaultdict[int, List]] = defaultdict(lambda: defaultdict(list))
        predicates_metadata = {}

        for entry in template:
            if isinstance(entry, Rule):
                weight_indices: List[Optional[int]] = []
                if isinstance(entry.head, WeightedRelation) and entry.head.weight is not None:
                    weight_indices.append(weight_index)
                    weight_index += 1
                else:
                    weight_indices.append(None)
                for body_relation in entry.body:
                    if isinstance(body_relation, WeightedRelation) and body_relation.weight is not None:
                        weight_indices.append(weight_index)
                        weight_index += 1
                    else:
                        weight_indices.append(None)
                batched_relations[entry.head.predicate.name][entry.head.predicate.arity].append((entry, weight_indices))
            elif isinstance(entry, BaseRelation):
                if isinstance(entry, WeightedRelation) and entry.weight is not None:
                    weight_indices = [weight_index]
                    weight_index += 1
                else:
                    weight_indices = [None]
                batched_relations[entry.predicate.name][entry.predicate.arity].append((entry, weight_indices))
            elif isinstance(entry, PredicateMetadata):
                predicates_metadata[str(entry.predicate)] = entry.metadata
            else:
                raise NotImplementedError("Template can contain only relations or predicate metadata!")
        return batched_relations, predicates_metadata

    def _convert(self):
        weights = self.model.state_dict()["weights"]

        rule_default_activation = str(self.settings.rule_transformation).lower()
        relation_default_activation = str(self.settings.relation_transformation).lower()
        default_aggregation = str(Aggregation.AVG).lower()

        batched_relations, predicates_metadata = self._process_template_entries()

        sql_source_headers = []
        sql_source = []

        for name, arities in batched_relations.items():
            for arity, relations_by_arity in arities.items():
                if f"{name}/{arity}" in self.table_mappings:
                    raise Exception

                is_fact = False

                for index, (relation, weight_indices) in enumerate(relations_by_arity):
                    if isinstance(relation, Rule):
                        act, agg = rule_default_activation, default_aggregation

                        if relation.metadata is not None:
                            if relation.metadata.transformation is not None:
                                act = str(relation.metadata.transformation).lower()
                            if relation.metadata.aggregation is not None:
                                agg = str(relation.metadata.aggregation).lower()

                        self._used_functions.add(act)
                        self._used_functions.add(agg)

                        sql_func = self.get_rule_sql_function(relation, index, act, agg, weight_indices, weights)
                    else:
                        is_fact = True
                        sql_func = self.get_fact_sql_function(relation, index, weight_indices, weights)
                    sql_source.append(sql_func)

                activation = relation_default_activation
                aggregation = default_aggregation
                predicate_metadata = predicates_metadata.get(f"{name}/{arity}", None)

                if predicate_metadata is not None:
                    if predicate_metadata.transformation is not None:
                        activation = str(predicate_metadata.transformation).lower()
                    if predicate_metadata.aggregation is not None:
                        aggregation = str(predicate_metadata.aggregation).lower()

                self._used_functions.add(activation)
                self._used_functions.add(aggregation)

                sql_func = self.get_rule_aggregation_function(
                    name,
                    arity,
                    len(relations_by_arity),
                    activation,
                    aggregation,
                    is_fact,
                )

                sql_source.append(sql_func)

                sql_funcs = self.get_relation_interface_sql_function(name, arity)
                sql_source_headers.append(sql_funcs[0])
                sql_source.append(sql_funcs[1])

        self.std_functions = self.get_helpers(self._used_functions)

        sql_source_headers.extend(sql_source)
        self.sql_source = "\n".join(sql_source_headers)

    def get_relation_interface_sql_function(self, relation: str, arity: int) -> Tuple[str, str]:
        raise NotImplementedError

    def get_rule_sql_function(
        self, rule: Rule, index: int, activation: str, aggregation: str, weight_indices: List[int], weights
    ) -> str:
        raise NotImplementedError

    def get_fact_sql_function(self, relation: BaseRelation, index: int, weight_indices: List[int], weights) -> str:
        raise NotImplementedError

    def get_rule_aggregation_function(
        self, name: str, arity: int, number_of_rules: int, activation: str, aggregation: str, is_fact: bool = False
    ) -> str:
        raise NotImplementedError

    def get_helpers(self, functions: Set[str]) -> str:
        raise NotImplementedError

    def get_std_functions(self) -> str:
        if self.sql_source is None:
            self._convert()
        if self.sql_source is None:
            return ""

        return self.std_functions

    def to_sql(self) -> str:
        if self.sql_source is None:
            self._convert()
        if self.sql_source is None:
            return ""

        return self.sql_source

    @staticmethod
    def _is_var(term) -> bool:
        """Helper check if term is a variable or constant"""
        return str(term)[0].isupper()
