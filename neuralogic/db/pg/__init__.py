from typing import List, Set

from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.db.convertor import Convertor
from neuralogic.db.pg.helpers import helpers


class PostgresConvertor(Convertor):
    FUNCTION_MAP = {
        "mul": "pynelo_mul",
        "sum": "pynelo_sum",
        "tanh": "pynelo_tanh",
        "sigmoid": "pynelo_sigmoid",
        "relu": "pynelo_relu",
        "identity": "",
        "avg": "AVG",
        "max": "MAX",
        "min": "MIN",
    }

    def get_helpers(self, functions: Set[str]) -> str:
        used_functions = {fun for fun in functions}
        used_functions.add("mul")
        used_functions.add("sum")

        return "\n".join(helpers[fun] for fun in used_functions if helpers[fun])

    def get_fact_sql_function(self, relation: BaseRelation, index: int, weight_indices: List[int], weights) -> str:
        """Generate a SQL function for a ground fact"""
        return_type = "".join(f", t{i} TEXT" for i in range(len(relation.terms)))

        if weight_indices[0] is None:
            value = 1
        else:
            value = weights[weight_indices[0]]
        select = ", ".join(f"'{term}' as t{i}" for i, term in enumerate(relation.terms))

        return f"""
CREATE OR REPLACE FUNCTION __{relation.predicate.name}_{relation.predicate.arity}_{index}()
RETURNS Table(value NUMERIC{return_type})
AS $$
    SELECT {value} as value{', ' if select else ''}{select}
$$ LANGUAGE SQL STABLE;
            """.strip()

    def get_relation_interface_sql_function(self, relation: str, arity: int) -> str:
        """Return the SQL function that should by used by the end users"""
        terms = ", ".join(f"p{i} TEXT" for i in range(arity))
        where = [f"out.t{i} = p{i}" for i in range(arity)]
        where_clause = f" WHERE {' AND '.join(where)}"

        return f"""
CREATE OR REPLACE FUNCTION {relation}({terms}) RETURNS Table(value NUMERIC) AS $$
    SELECT out.value FROM __{relation}_{arity}() as out{'' if not where else where_clause}
$$ LANGUAGE SQL STABLE;
            """.strip()

    def get_rule_aggregation_function(
        self, name: str, arity: int, number_of_rules: int, activation: str, aggregation: str
    ) -> str:
        """
        Generete SQL function which aggregates rule functions (something like the aggregation neuron)
        """
        inner_select = []
        for i in range(arity):
            selects = (f"s{index}.t{i}" for index in range(number_of_rules))
            inner_select.append(f"COALESCE({','.join(selects)}) as t{i}")

        # inner_select = [f"s0.t{i} as t{i}" for i in range(arity)]
        inner_value_select = []
        from_clause = []

        for index in range(number_of_rules):
            join_on = [f"s{index}.t{i} = s{index - 1}.t{i}" for i in range(arity)]
            inner_value_select.append(f"COALESCE(s{index}.value, 0)")

            if len(inner_value_select) == 2:
                inner_value_select = [f"pynelo_sum({', '.join(inner_value_select)})"]

            function_name = f"__{name}_{arity}_{index}()"

            if not from_clause:
                from_clause.append(f"{function_name} as s{index}")
            else:
                from_clause.append(
                    f"{function_name} AS s{index} ON {'1 = 1' if not join_on else ' AND '.join(join_on)}"
                )

        if len(inner_value_select) == 1:
            inner_select.append(f"{activation}({inner_value_select[0]}) as value")

        select = [f"{aggregation}(out.value) as value"]
        select.extend(f"out.t{i}" for i in range(arity))
        select = ", ".join(select)

        from_clause = f"{' FULL OUTER JOIN '.join(from_clause)}"
        group_by_clause = f" GROUP BY {', '.join('out.t' + str(v) for v in range(arity))}"
        from_clause = f"SELECT {', '.join(inner_select)} FROM {from_clause}"
        return_type = "".join(f", t{i} TEXT" for i in range(arity))

        return f"""
CREATE OR REPLACE FUNCTION __{name}_{arity}()
    RETURNS Table(value NUMERIC{return_type})
    AS $$
    SELECT {select}
    FROM ({from_clause}) AS out{'' if arity == 0 else group_by_clause}
$$ LANGUAGE SQL STABLE;
            """.strip()

    def get_rule_sql_function(
        self, rule: Rule, index: int, activation: str, aggregation: str, weight_indices: List[int], weights
    ) -> str:
        """Return the SQL function of one rule"""
        if weight_indices[0] is None:
            select = [f"{aggregation}(out.value) as value"]
        else:
            select = [f"{aggregation}(pynelo_mul({weights[weight_indices[0]]}, out.value)) as value"]

        vars_mapping = {}
        where = []
        vars_body_mapping = {}
        inner_select = []
        inner_value_select = []
        from_clause = []

        for index, term in enumerate(rule.head.terms):
            if Convertor._is_var(term):
                term_name = f"t{index}"
                vars_mapping[str(term)] = term_name

                select.append(f"out.{term_name} as {term_name}")
            else:
                select.append(f"'{term}' as t{index}")

        for t_index, (relation, weight_id) in enumerate(zip(rule.body, weight_indices[1:])):
            join_on = []

            relation_mapping = self.table_mappings.get(str(relation.predicate), None)
            value = "value" if relation_mapping is None else relation_mapping.value_column
            selected_value = 1 if value is None else f"s{t_index}.{value}"

            if weight_id is None:
                inner_value_select.append(selected_value)
            else:
                inner_value_select.append(f"pynelo_mul({weights[weight_id]}, {selected_value})")

            if len(inner_value_select) == 2:
                inner_value_select = [f"pynelo_sum({', '.join(inner_value_select)})"]

            for index, term in enumerate(relation.terms):
                if relation_mapping is None:
                    field = f"t{index}"
                else:
                    field = relation_mapping.term_columns[index]

                if not self._is_var(term):
                    where.append(f"s{t_index}.{field} = '{str(term)}'")
                    continue
                if str(term) in vars_body_mapping:
                    join_on.append(f"s{t_index}.{field} = {vars_body_mapping[str(term)]}")
                    continue
                vars_body_mapping[str(term)] = f"s{t_index}.{field}"
                if str(term) in vars_mapping:
                    inner_select.append(f"s{t_index}.{field} AS {vars_mapping[str(term)]}")

            if relation_mapping is None:
                function_name = f"__{relation.predicate.name}_{relation.predicate.arity}()"
            else:
                function_name = relation_mapping.table_name

            if not from_clause:
                from_clause.append(f"{function_name} AS s{t_index}")
            else:
                from_clause.append(
                    f"{function_name} AS s{t_index} ON {'1 = 1' if not join_on else ' AND '.join(join_on)}"
                )

        from_clause = f"{' INNER JOIN '.join(from_clause)}"
        where_clause = f" WHERE {' AND '.join(where)}"
        group_by_clause = f" GROUP BY {', '.join('out.' + v for v in vars_mapping.values())}"

        if len(inner_value_select) == 1:
            inner_select.append(f"{activation}({inner_value_select[0]}) as value")

        from_clause = f"SELECT {', '.join(inner_select)} FROM {from_clause}{'' if not where else where_clause}"
        return_type = "".join(f", t{i} TEXT" for i in range(len(rule.head.terms)))

        return f"""
CREATE OR REPLACE FUNCTION __{rule.head.predicate.name}_{rule.head.predicate.arity}_{index}()
RETURNS Table(value NUMERIC{return_type})
AS $$
    SELECT {', '.join(select)}
    FROM ({from_clause}) AS out{'' if not vars_mapping else group_by_clause}
$$ LANGUAGE SQL STABLE;
            """.strip()
