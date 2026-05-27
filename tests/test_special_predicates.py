import pytest

from neuralogic.core import Model, V, R
from neuralogic.dataset import Dataset, Sample


def test_neq():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.neq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 2

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "1"

    assert out[1]["X"] == "1"
    assert out[1]["Y"] == "-1"


def test_leq():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.leq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 3

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "-1"

    assert out[1]["X"] == "-1"
    assert out[1]["Y"] == "1"

    assert out[2]["X"] == "1"
    assert out[2]["Y"] == "1"


def test_geq():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.geq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 3

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "-1"

    assert out[1]["X"] == "1"
    assert out[1]["Y"] == "-1"

    assert out[2]["X"] == "1"
    assert out[2]["Y"] == "1"


def test_lt():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.lt(V.X, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 1

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "1"


def test_gt():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.gt(V.X, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 1

    assert out[0]["X"] == "1"
    assert out[0]["Y"] == "-1"


def test_eq():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.eq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 2

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "-1"

    assert out[1]["X"] == "1"
    assert out[1]["Y"] == "1"


def test_next():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.next(V.X, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1), R.val(0)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 2

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "0"

    assert out[1]["X"] == "0"
    assert out[1]["Y"] == "1"


def test_next_skip():
    model = Model()
    model += R.head(V.X, V.Y) <= (R.special.next(V.X, V.Z), R.special.next(V.Z, V.Y), R.val(V.X), R.val(V.Y))

    examples = [R.val(1), R.val(-1), R.val(0)]

    out = model.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 1

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "1"


def test_alldiff():
    model = Model()
    model += R.head(V.X, V.Y, V.Z) <= (R.special.alldiff(V.X, V.Y, V.Z), R.val(V.X), R.val(V.Y), R.val(V.Z))

    examples = [R.val(1), R.val(-1), R.val(0)]

    out = model.q(R.head(V.X, V.Y, V.Z), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 6


@pytest.mark.parametrize(
    "predicate,expected",
    (
        (R.special.add_eval, 3 + 2),
        (R.special.sub_eval, 3 - 2),
        (R.special.div_eval, 3 / 2),
        (R.special.mul_eval, 3 * 2),
        (R.special.mod_eval, 3 % 2),
        (R.special.max_eval, max(3, 2)),
        (R.special.min_eval, min(3, 2)),
    ),
)
def test_eval_predicates(predicate, expected):
    var_value, const_value = 3, 2

    model = Model()
    model += R.head(V.X) <= (predicate(V.X, const_value))

    m = model.build()

    dataset = Dataset([Sample(R.head(var_value), [R.val(var_value)])])

    built_dataset = m.build_dataset(dataset)
    res = m(built_dataset)

    assert res[0] == expected
