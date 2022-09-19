from neuralogic.core import Template, V, R
from neuralogic.inference import InferenceEngine


def test_neq():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.neq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 2

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "1"

    assert out[1]["X"] == "1"
    assert out[1]["Y"] == "-1"


def test_leq():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.leq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 3

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "-1"

    assert out[1]["X"] == "-1"
    assert out[1]["Y"] == "1"

    assert out[2]["X"] == "1"
    assert out[2]["Y"] == "1"


def test_geq():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.geq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 3

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "-1"

    assert out[1]["X"] == "1"
    assert out[1]["Y"] == "-1"

    assert out[2]["X"] == "1"
    assert out[2]["Y"] == "1"


def test_lt():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.lt(V.X, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 1

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "1"


def test_gt():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.gt(V.X, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 1

    assert out[0]["X"] == "1"
    assert out[0]["Y"] == "-1"


def test_eq():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.eq(V.X, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 2

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "-1"

    assert out[1]["X"] == "1"
    assert out[1]["Y"] == "1"


def test_next():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.next(V.X, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1), R.val(0)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 2

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "0"

    assert out[1]["X"] == "0"
    assert out[1]["Y"] == "1"


def test_next_skip():
    template = Template()
    template += R.head(V.X, V.Y) <= (R.special.next(V.X, V.Z), R.special.next(V.Z, V.Y), R.val(V.X), R.val(V.Y))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1), R.val(0)]

    out = inference_engine.q(R.head(V.X, V.Y), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 1

    assert out[0]["X"] == "-1"
    assert out[0]["Y"] == "1"


def test_alldiff():
    template = Template()
    template += R.head(V.X, V.Y, V.Z) <= (R.special.alldiff(V.X, V.Y, V.Z), R.val(V.X), R.val(V.Y), R.val(V.Z))

    inference_engine = InferenceEngine(template)
    examples = [R.val(1), R.val(-1), R.val(0)]

    out = inference_engine.q(R.head(V.X, V.Y, V.Z), examples)
    out = sorted(list(out), key=lambda a: a["X"] + a["Y"])

    assert len(out) == 6
