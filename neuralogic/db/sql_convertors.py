from typing import List, Dict


def _is_var(term) -> bool:
    """Helper check if term is a variable or constant"""
    return str(term)[0].isupper()


def _get_rule_aggregation_sql_function(rule: str, arity: int, number_of_rules: int, activation: str, aggregation: str):
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

        function_name = f"__{rule}_{arity}_{index}()"

        if not from_clause:
            from_clause.append(f"{function_name} as s{index}")
        else:
            from_clause.append(f"{function_name} AS s{index} ON {'1 = 1' if not join_on else ' AND '.join(join_on)}")

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
CREATE OR REPLACE FUNCTION __{rule}_{arity}()
    RETURNS Table(value NUMERIC{return_type})
    AS $$
    SELECT {select}
    FROM ({from_clause}) AS out{'' if arity == 0 else group_by_clause}
$$ LANGUAGE SQL STABLE;
    """.strip()


def _get_relation_interface_sql_function(rule: str, arity: int) -> str:
    """Return the SQL function that should by used by the end users"""
    terms = ", ".join(f"p{i} TEXT" for i in range(arity))
    where = [f"out.t{i} = p{i}" for i in range(arity)]
    where_clause = f" WHERE {' AND '.join(where)}"

    return f"""
CREATE OR REPLACE FUNCTION {rule}({terms}) RETURNS Table(value NUMERIC) AS $$
    SELECT out.value FROM __{rule}_{arity}() as out{'' if not where else where_clause}
$$ LANGUAGE SQL STABLE;
    """.strip()


def _get_rule_sql_function(rule, id, activation, aggregation, weights_ids: List[int], weights: Dict):
    """Return the SQL function of one rule"""
    if weights_ids[0] is None:
        select = [f"{aggregation}(out.value) as value"]
    else:
        select = [f"{aggregation}(pynelo_mul({weights[weights_ids[0]]}, out.value)) as value"]

    vars_mapping = {}
    where = []
    vars_body_mapping = {}
    inner_select = []
    inner_value_select = []
    from_clause = []

    for index, term in enumerate(rule.head.terms):
        if _is_var(term):
            term_name = f"t{index}"
            vars_mapping[str(term)] = term_name

            select.append(f"out.{term_name} as {term_name}")
        else:
            select.append(f"'{term}' as t{index}")

    for t_index, (relation, weight_id) in enumerate(zip(rule.body, weights_ids[1:])):
        join_on = []

        if weight_id is None:
            inner_value_select.append(f"s{t_index}.value")
        else:
            inner_value_select.append(f"pynelo_mul({weights[weight_id]}, s{t_index}.value)")

        if len(inner_value_select) == 2:
            inner_value_select = [f"pynelo_sum({', '.join(inner_value_select)})"]

        for index, term in enumerate(relation.terms):
            if _is_var(term):
                if str(term) in vars_body_mapping:
                    join_on.append(f"s{t_index}.t{index} = {vars_body_mapping[str(term)]}")
                else:
                    vars_body_mapping[str(term)] = f"s{t_index}.t{index}"
                    if str(term) in vars_mapping:
                        inner_select.append(f"s{t_index}.t{index} AS {vars_mapping[str(term)]}")
            else:
                where.append(f"s{t_index}.t{index} = '{str(term)}'")

        function_name = f"__{relation.predicate.name}_{relation.predicate.arity}"

        if not from_clause:
            from_clause.append(f"{function_name}() AS s{t_index}")
        else:
            from_clause.append(
                f"{function_name}() AS s{t_index} ON {'1 = 1' if not join_on else ' AND '.join(join_on)}"
            )

    from_clause = f"{' INNER JOIN '.join(from_clause)}"
    where_clause = f" WHERE {' AND '.join(where)}"
    group_by_clause = f" GROUP BY {', '.join('out.' + v for v in vars_mapping.values())}"

    if len(inner_value_select) == 1:
        inner_select.append(f"{activation}({inner_value_select[0]}) as value")

    from_clause = f"SELECT {', '.join(inner_select)} FROM {from_clause}{'' if not where else where_clause}"
    return_type = "".join(f", t{i} TEXT" for i in range(len(rule.head.terms)))

    return f"""
CREATE OR REPLACE FUNCTION __{rule.head.predicate.name}_{rule.head.predicate.arity}_{id}()
RETURNS Table(value NUMERIC{return_type})
AS $$
    SELECT {', '.join(select)}
    FROM ({from_clause}) AS out{'' if not vars_mapping else group_by_clause}
$$ LANGUAGE SQL STABLE;
    """.strip()


def _get_fact_sql_function(relation, id, weights_ids: List[int], weights: Dict):
    """Generate a SQL function for a ground fact"""
    return_type = "".join(f", t{i} TEXT" for i in range(len(relation.terms)))

    if weights_ids[0] is None:
        value = 1
    else:
        value = weights[weights_ids[0]]
    select = ", ".join(f"'{term}' as t{i}" for i, term in enumerate(relation.terms))

    return f"""
CREATE OR REPLACE FUNCTION __{relation.predicate.name}_{relation.predicate.arity}_{id}()
RETURNS Table(value NUMERIC{return_type})
AS $$
    SELECT {value} as value{', ' if select else ''}{select}
$$ LANGUAGE SQL STABLE;
    """.strip()
