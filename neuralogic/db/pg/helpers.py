MUL = """
CREATE FUNCTION neuralogic_std.mul(in_a NUMERIC, in_b NUMERIC) RETURNS NUMERIC AS
$$
SELECT (in_a * in_b)::NUMERIC
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.mul(in_a NUMERIC, in_b NUMERIC[]) RETURNS NUMERIC[] AS
$$
SELECT array_agg(in_a * un_b) FROM unnest(in_b) as un_b
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.mul(in_b NUMERIC[], in_a NUMERIC) RETURNS NUMERIC[] AS
$$
SELECT neuralogic_std.mul(in_a, in_b)
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.mul(in_a NUMERIC[], in_b NUMERIC[]) RETURNS NUMERIC AS
$$
SELECT SUM(un_a * un_b)::NUMERIC FROM unnest(in_a, in_b) as t(un_a, un_b)
$$
LANGUAGE SQL IMMUTABLE;
"""

SUM = """

CREATE FUNCTION neuralogic_std.sum(in_a NUMERIC, in_b NUMERIC) RETURNS NUMERIC AS
$$
SELECT (in_a + in_b)::NUMERIC
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.sum(in_a NUMERIC, in_b NUMERIC[]) RETURNS NUMERIC[] AS
$$
SELECT array_agg(in_a + un_b) FROM unnest(in_b) as un_b
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.sum(in_b NUMERIC[], in_a NUMERIC) RETURNS NUMERIC[] AS
$$
SELECT neuralogic_std.sum(in_a, in_b)
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.sum(in_a NUMERIC[], in_b NUMERIC[]) RETURNS NUMERIC AS
$$
SELECT SUM(un_a + un_b)::NUMERIC FROM unnest(in_a, in_b) as t(un_a, un_b)
$$
LANGUAGE SQL IMMUTABLE;
"""

TANH = """
CREATE FUNCTION neuralogic_std.tanh(in_a NUMERIC[]) RETURNS NUMERIC[] AS
$$
SELECT array_agg(tanh(un_a)) FROM unnest(in_a) as un_a
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.tanh(in_a NUMERIC) RETURNS NUMERIC AS
$$
SELECT tanh(in_a)
$$
LANGUAGE SQL IMMUTABLE;
"""

SIGMOID = """
CREATE FUNCTION neuralogic_std.sigmoid(in_a NUMERIC) RETURNS NUMERIC AS
$$
SELECT 1 / (exp(-in_a) + 1)
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.sigmoid(in_a NUMERIC[]) RETURNS NUMERIC[] AS
$$
SELECT array_agg(neuralogic_std.sigmoid(un_a)) FROM unnest(in_a) as un_a
$$
LANGUAGE SQL IMMUTABLE;
"""

RELU = """
CREATE FUNCTION neuralogic_std.relu(in_a NUMERIC) RETURNS NUMERIC AS
$$
SELECT CASE WHEN in_a > 0 THEN in_a ELSE 0 END
$$
LANGUAGE SQL IMMUTABLE;

CREATE FUNCTION neuralogic_std.relu(in_a NUMERIC[]) RETURNS NUMERIC[] AS
$$
SELECT array_agg(neuralogic_std.relu(un_a)) FROM unnest(in_a) as un_a
$$
LANGUAGE SQL IMMUTABLE;
"""


helpers = {
    "mul": MUL,
    "sum": SUM,
    "tanh": TANH,
    "sigmoid": SIGMOID,
    "relu": RELU,
    "identity": None,
    "avg": None,
    "max": None,
    "min": None,
}
