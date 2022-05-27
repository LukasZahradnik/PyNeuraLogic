from typing import List, Set, Tuple

from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule
from neuralogic.db.convertor import Convertor
from neuralogic.db.pg.helpers import helpers


FUNCTION_TEMPLATE = """
CREATE OR REPLACE FUNCTION {name}({params}) RETURNS {return_type} AS $$
{body}
$$ LANGUAGE SQL STABLE;
""".strip()


FUNCTION_MAP = {
    "mul": "neuralogic_std.mul",
    "sum": "neuralogic_std.sum",
    "tanh": "neuralogic_std.tanh",
    "sigmoid": "neuralogic_std.sigmoid",
    "relu": "neuralogic_std.relu",
    "identity": "",
    "avg": "AVG",
    "max": "MAX",
    "min": "MIN",
}


class PostgresConvertor(Convertor):
    @staticmethod
    def get_function(name: str, params: List[str], return_type: List[str], body: str) -> str:
        return FUNCTION_TEMPLATE.format(
            name=name, params=",".join(params), return_type=f"Table({','.join(return_type)})", body=body
        )

    @staticmethod
    def get_empty_function(name: str, params: List[str], return_type: List[str]) -> str:
        if len(return_type) == 0:
            raise NotImplementedError

        select = ["1"]
        for _ in return_type[1:]:
            select.append("'1'")

        return PostgresConvertor.get_function(name, params, return_type, f"SELECT {','.join(select)}")

    def get_helpers(self, functions: Set[str]) -> str:
        used_functions = {fun for fun in functions}
        used_functions.add("mul")
        used_functions.add("sum")

        function_sources = ["CREATE SCHEMA IF NOT EXISTS neuralogic_std;" "CREATE SCHEMA IF NOT EXISTS neuralogic;"]

        for fun in used_functions:
            if helpers[fun]:
                function_sources.append(helpers[fun].strip())
        return "\n".join(function_sources)

    def get_fact_sql_function(
        self, relation: BaseRelation, index: int, weight_indices: List[int], weights
    ) -> Tuple[str, str]:
        """Generate a SQL function for a ground fact"""
        value = 1 if weight_indices[0] is None else weights[weight_indices[0]]

        return_type = ["value NUMERIC", *(f"t{i}" for i in range(len(relation.terms)))]
        select = [f"{value} as value", *(f"'{term}' as t{i}" for i, term in enumerate(relation.terms))]

        name = f"neuralogic._{relation.predicate.name}_{relation.predicate.arity}_{index}"

        return (
            self.get_empty_function(name, [], return_type),
            self.get_function(name, [], return_type, f"SELECT {select}"),
        )

    def get_relation_interface_sql_function(self, relation: str, arity: int) -> Tuple[str, str]:
        """Return the SQL function that should by used by the end users"""
        params = [f"p{i} TEXT" for i in range(arity)]
        where = [f"out.t{i} = p{i}" for i in range(arity)]
        where_clause = f" WHERE {' AND '.join(where)}"

        name = f"neuralogic.{relation}"
        return_type = ["value NUMERIC"]
        body = f"SELECT out.value FROM neuralogic._{relation}_{arity}() as out{'' if not where else where_clause}"

        return (
            self.get_empty_function(name, params, return_type),
            self.get_function(name, params, return_type, body),
        )

    def get_rule_aggregation_function(
        self, name: str, arity: int, number_of_rules: int, activation: str, aggregation: str
    ) -> Tuple[str, str]:
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
                inner_value_select = [f"{FUNCTION_MAP['sum']}({', '.join(inner_value_select)})"]

            function_name = f"neuralogic._{name}_{arity}_{index}()"

            if not from_clause:
                from_clause.append(f"{function_name} as s{index}")
            else:
                from_clause.append(
                    f"{function_name} AS s{index} ON {'1 = 1' if not join_on else ' AND '.join(join_on)}"
                )

        if len(inner_value_select) == 1:
            inner_select.append(f"{FUNCTION_MAP[activation]}({inner_value_select[0]}) as value")

        select = [f"{FUNCTION_MAP[aggregation]}(out.value) as value"]
        select.extend(f"out.t{i}" for i in range(arity))
        select = ", ".join(select)

        from_clause = f"{' FULL OUTER JOIN '.join(from_clause)}"
        group_by_clause = f" GROUP BY {', '.join('out.t' + str(v) for v in range(arity))}"
        from_clause = f"SELECT {', '.join(inner_select)} FROM {from_clause}"

        return_type = ["value NUMERIC", *(f"t{i} TEXT" for i in range(arity))]
        name = f"neuralogic._{name}_{arity}"
        body = f"SELECT {select} FROM ({from_clause}) AS out{'' if arity == 0 else group_by_clause}"

        return (
            self.get_empty_function(name, [], return_type),
            self.get_function(name, [], return_type, body),
        )

    def get_rule_sql_function(
        self, rule: Rule, index: int, activation: str, aggregation: str, weight_indices: List[int], weights
    ) -> Tuple[str, str]:
        """Return the SQL function of one rule"""
        if weight_indices[0] is None:
            select = [f"{FUNCTION_MAP[aggregation]}(out.value) as value"]
        else:
            select = [
                f"{FUNCTION_MAP[aggregation]}({FUNCTION_MAP['mul']}"
                f"({weights[weight_indices[0]]}, out.value)) as value"
            ]

        vars_mapping = {}
        where = []
        vars_body_mapping = {}
        inner_select = []
        inner_value_select = []
        from_clause = []

        for term_idx, term in enumerate(rule.head.terms):
            if Convertor._is_var(term):
                term_name = f"t{term_idx}"
                vars_mapping[str(term)] = term_name

                select.append(f"out.{term_name} as {term_name}")
            else:
                select.append(f"'{term}' as t{term_idx}")

        for t_index, (relation, weight_id) in enumerate(zip(rule.body, weight_indices[1:])):
            join_on = []

            relation_mapping = self.table_mappings.get(str(relation.predicate), None)
            value = "value" if relation_mapping is None else relation_mapping.value_column
            selected_value = 1 if value is None else f"s{t_index}.{value}"

            if weight_id is None:
                inner_value_select.append(selected_value)
            else:
                inner_value_select.append(f"{FUNCTION_MAP['mul']}({weights[weight_id]}, {selected_value})")

            if len(inner_value_select) == 2:
                inner_value_select = [f"{FUNCTION_MAP['sum']}({', '.join(inner_value_select)})"]

            for term_idx, term in enumerate(relation.terms):
                if relation_mapping is None:
                    field = f"t{term_idx}"
                else:
                    field = relation_mapping.term_columns[term_idx]

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
                function_name = f"neuralogic._{relation.predicate.name}_{relation.predicate.arity}()"
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
            inner_select.append(f"{FUNCTION_MAP[activation]}({inner_value_select[0]}) as value")

        from_clause = f"SELECT {', '.join(inner_select)} FROM {from_clause}{'' if not where else where_clause}"

        return_type = ["value NUMERIC", *(f"t{i} TEXT" for i in range(len(rule.head.terms)))]

        name = f"neuralogic._{rule.head.predicate.name}_{rule.head.predicate.arity}_{index}"
        body = f"SELECT {', '.join(select)} FROM ({from_clause}) AS out{'' if not vars_mapping else group_by_clause}"

        return (
            self.get_empty_function(name, [], return_type),
            self.get_function(name, [], return_type, body),
        )
