import pytest
import torch

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.1
INPUT = 0.7
START = 0.9
TARGET = 1.0


def _built(transformation, error):
    """out = transformation(w * input), so both the value and its slope come from one weight.

    The transformation is stated on the queried head, which output-function inference leaves alone - so this
    also stands as a check that it still does.
    """
    model = Model()
    model += R.source("a")[INPUT].fixed()
    model += (R.out(V.X)["w":1, 1] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [transformation]
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=error,
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = START
    built.load_state_dict(state)
    return built, index, built.build_dataset(Dataset([Sample(R.out("a")[TARGET], [R.exists("a")])]))


def _value_and_gradient(transformation, error):
    built, index, dataset = _built(transformation, error)
    value = float(built(dataset)[0])
    before = float(built.state_dict()["weights"][index])
    built.train(dataset, epochs=1)
    gradient = (before - float(built.state_dict()["weights"][index])) / LEARNING_RATE
    return value, gradient


def _torch_value_and_gradient(activation, loss):
    weight = torch.nn.Parameter(torch.tensor([START], dtype=torch.float64))
    output = activation(weight * INPUT)
    loss(output, torch.tensor([TARGET], dtype=torch.float64)).sum().backward()
    return float(output.item()), float(weight.grad.item())


@pytest.mark.parametrize(
    "transformation, activation",
    [
        (Transformation.IDENTITY, lambda x: x),
        (Transformation.SIGMOID, torch.sigmoid),
        (Transformation.TANH, torch.tanh),
        (Transformation.RELU, torch.relu),
        (Transformation.EXP, torch.exp),
        (Transformation.SQRT, torch.sqrt),
        (Transformation.LOG, torch.log),
        (Transformation.INVERSE, lambda x: 1 / x),
        (Transformation.LEAKY_RELU, lambda x: torch.nn.functional.leaky_relu(x, 0.01)),
        (Transformation.SIGNUM, torch.sign),
    ],
    ids=["identity", "sigmoid", "tanh", "relu", "exp", "sqrt", "log", "inverse", "leaky_relu", "signum"],
)
def test_transformation_matches_torch(transformation, activation):
    """An activation has to agree with Torch in value and in what it does to the gradient passing through."""
    value, gradient = _value_and_gradient(transformation, MSE(reduction="sum"))
    expected_value, expected_gradient = _torch_value_and_gradient(
        activation, lambda out, target: (out - target) ** 2
    )

    assert value == pytest.approx(expected_value, abs=1e-9)
    assert gradient == pytest.approx(expected_gradient, abs=1e-9)


@pytest.mark.parametrize(
    "error, loss",
    [
        (MSE(reduction="sum"), lambda out, target: (out - target) ** 2),
        (
            CrossEntropy(with_logits=False, reduction="sum"),
            lambda out, target: torch.nn.functional.binary_cross_entropy(out, target),
        ),
    ],
    ids=["squared_diff", "cross_entropy"],
)
def test_error_function_matches_torch(error, loss):
    """The loss has to agree with Torch in the gradient it sends back, which is what training sees of it.

    A squashed output on both sides, so cross-entropy is compared where it is defined - and so the
    with_logits=False convention is pinned rather than assumed.
    """
    _, gradient = _value_and_gradient(Transformation.SIGMOID, error)
    _, expected = _torch_value_and_gradient(torch.sigmoid, loss)

    assert gradient == pytest.approx(expected, abs=1e-9)


def test_the_error_sums_over_a_vector_output_rather_than_averaging():
    """MSE here adds the squared differences up; torch.nn.MSELoss divides by how many there were.

    A scalar output cannot tell the two apart, and neither can a comparison run on Adam - its step follows
    the sign of the gradient far more than the size, so it agrees with either convention to about 1e-7.
    Under plain SGD they are a factor of the output width apart, which is what this pins.
    """
    width = 3
    inputs = [0.7, -0.4, 0.2]
    weight = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.6], [-0.1, 0.2, 0.8]]
    target = [1.0, 0.0, -0.5]

    model = Model()
    model += R.source("a")[inputs].fixed()
    model += (R.out(V.X) <= R.source(V.X)["w":width, width]) | [Combination.SUM, Transformation.IDENTITY]
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = weight
    built.load_state_dict(state)
    dataset = built.build_dataset(Dataset([Sample(R.out("a")[target], [R.exists("a")])]))

    before = built.state_dict()["weights"][index]
    built.train(dataset, epochs=1)
    after = built.state_dict()["weights"][index]
    gradient = [(b - a) / LEARNING_RATE for rb, ra in zip(before, after) for b, a in zip(rb, ra)]

    summed = _torch_matrix_gradient(inputs, weight, target, lambda diff: diff.sum())
    averaged = _torch_matrix_gradient(inputs, weight, target, lambda diff: diff.mean())

    assert gradient == pytest.approx(summed, abs=1e-9)
    assert max(abs(ours - theirs) for ours, theirs in zip(gradient, averaged)) > 1e-3


def _torch_matrix_gradient(inputs, weight, target, reduce):
    parameter = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float64))
    output = parameter @ torch.tensor(inputs, dtype=torch.float64)
    reduce((output - torch.tensor(target, dtype=torch.float64)) ** 2).backward()
    return [entry for row in parameter.grad.tolist() for entry in row]


VECTOR_INPUT = [0.7, -0.4, 0.2]
VECTOR_WEIGHT = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.6], [-0.1, 0.2, 0.8]]
VECTOR_TARGET = [1.0, 0.0, -0.5]


def _vector_value_and_gradient(transformation, error=None, target=None):
    """out = transformation(W @ x) over a whole vector at once, stated on the queried head."""
    model = Model()
    model += R.source("a")[VECTOR_INPUT].fixed()
    model += (R.out(V.X) <= R.source(V.X)["w":3, 3]) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [transformation]
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=error if error is not None else MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = VECTOR_WEIGHT
    built.load_state_dict(state)
    dataset = built.build_dataset(
        Dataset([Sample(R.out("a")[target if target is not None else VECTOR_TARGET], [R.exists("a")])])
    )

    value = [float(v) for v in built(dataset)[0]]
    before = built.state_dict()["weights"][index]
    built.train(dataset, epochs=1)
    after = built.state_dict()["weights"][index]
    gradient = [(b - a) / LEARNING_RATE for rb, ra in zip(before, after) for b, a in zip(rb, ra)]
    return value, gradient


def test_softmax_matches_torch_including_its_off_diagonal_gradient():
    """Softmax is the first transformation here whose Jacobian is not diagonal.

    Every activation checked above acts on one number at a time, so a wrong derivative can only be wrong in
    that one place. Softmax mixes the whole vector, and each output depends on every input - so this is the
    one that says the engine carries a full Jacobian back rather than an elementwise slope.
    """
    value, gradient = _vector_value_and_gradient(Transformation.SOFTMAX)

    weight = torch.nn.Parameter(torch.tensor(VECTOR_WEIGHT, dtype=torch.float64))
    output = torch.softmax(weight @ torch.tensor(VECTOR_INPUT, dtype=torch.float64), dim=0)
    ((output - torch.tensor(VECTOR_TARGET, dtype=torch.float64)) ** 2).sum().backward()

    assert value == pytest.approx(output.tolist(), abs=1e-9)
    assert gradient == pytest.approx([e for row in weight.grad.tolist() for e in row], abs=1e-9)


PROBABILITY_TARGET = [0.6, 0.3, 0.1]


def test_cross_entropy_with_logits_matches_torch_on_a_scalar():
    """`with_logits=True` is SOFTENTROPY, which takes the raw logit and squashes inside the loss itself.

    The `with_logits=False` case above is compared behind a sigmoid; this is the other spelling, where
    nothing squashes the head and the loss is expected to. Getting the two the wrong way round produces a
    plausible number rather than an error, which is why both are pinned.
    """
    _, gradient = _value_and_gradient(Transformation.IDENTITY, CrossEntropy(with_logits=True, reduction="sum"))
    _, expected = _torch_value_and_gradient(
        lambda x: x,
        lambda out, target: torch.nn.functional.binary_cross_entropy_with_logits(out, target),
    )

    assert gradient == pytest.approx(expected, abs=1e-9)


def test_softmax_cross_entropy_over_a_vector_matches_torch():
    """Over a vector, SOFTENTROPY fuses the softmax into the loss - `target - softmax(logit)`.

    The source calls that a nice simplification of doing the two separately, and it is; but a fused form is
    exactly the kind that drifts from the composition it replaces without anything noticing. Compared here
    against torch spelling it out as `-(target * log_softmax(logit)).sum()`.
    """
    _, gradient = _vector_value_and_gradient(
        Transformation.IDENTITY, CrossEntropy(with_logits=True, reduction="sum"), target=PROBABILITY_TARGET
    )

    weight = torch.nn.Parameter(torch.tensor(VECTOR_WEIGHT, dtype=torch.float64))
    logit = weight @ torch.tensor(VECTOR_INPUT, dtype=torch.float64)
    loss = -(torch.tensor(PROBABILITY_TARGET, dtype=torch.float64) * torch.log_softmax(logit, 0)).sum()
    loss.backward()

    assert gradient == pytest.approx([e for row in weight.grad.tolist() for e in row], abs=1e-9)


def test_norm_matches_torch_layer_norm_including_its_off_diagonal_gradient():
    """NORM is layer normalisation, the second transformation here whose Jacobian is not diagonal.

    Every output depends on every input twice over, through the mean and through the variance, so an engine
    differentiating it one component at a time would be wrong in a different way than softmax would catch.
    Compared against torch's own `layer_norm` at the same `1e-10` epsilon, with no weight or bias, since the
    backend applies none.
    """
    value, gradient = _vector_value_and_gradient(Transformation.NORM)

    weight = torch.nn.Parameter(torch.tensor(VECTOR_WEIGHT, dtype=torch.float64))
    output = torch.nn.functional.layer_norm(
        weight @ torch.tensor(VECTOR_INPUT, dtype=torch.float64), (3,), eps=1e-10
    )
    ((output - torch.tensor(VECTOR_TARGET, dtype=torch.float64)) ** 2).sum().backward()

    assert value == pytest.approx(output.tolist(), abs=1e-9)
    assert gradient == pytest.approx([e for row in weight.grad.tolist() for e in row], abs=1e-9)


#: A scalar multiplied into a vector under a PRODUCT body, which is what an attention weight is
SCALAR_SIDE = 2.0
PRODUCT_INPUT = [0.5, -0.25, 0.75]
PRODUCT_WEIGHT = [[0.5, -0.2, 0.1], [0.4, -0.6, 0.25], [-0.1, 0.2, 0.8]]
PRODUCT_TARGET = [0.1, 0.2, -0.3]


def _scalar_times_vector(scalar_first: bool):
    """h(X) = s(X) * (W . v(X)), with the scalar written first or last in the body.

    The two spell the same function - a product does not care about order - so they have to step to the same
    weight. They did not: the derivative by a *scalar* input was taken as an outer product with the rest
    rather than a dot product, so writing the scalar first threw `scalar incrementBy by matrix` while writing
    it last worked. An attention weight is exactly a scalar multiplied into a vector message, which is how
    this surfaced.
    """
    model = Model()
    model += (R.s(V.X) <= R.a(V.X)) | [Transformation.IDENTITY]
    parts = [R.s(V.X), R.v(V.X)["w":3, 3]]
    model += (R.h(V.X) <= (parts if scalar_first else parts[::-1])) | [
        Combination.PRODUCT,
        Transformation.IDENTITY,
    ]
    model += R.h / 1 | [Transformation.IDENTITY]

    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = PRODUCT_WEIGHT
    built.load_state_dict(state)

    example = [R.a(1)[SCALAR_SIDE], R.v(1)[PRODUCT_INPUT]]
    data = built.build_dataset(Dataset([Sample(R.h(1)[PRODUCT_TARGET], example)]))
    value = [float(v) for v in built(data)[0]]
    built.train(data, epochs=1)
    return value, [entry for row in built.state_dict()["weights"][index] for entry in row]


@pytest.mark.parametrize("scalar_first", [True, False], ids=["scalar-first", "scalar-last"])
def test_scalar_times_vector_product_matches_torch(scalar_first):
    """Both body orders, each against torch, so neither a wrong shape nor a wrong sign can pass."""
    value, after = _scalar_times_vector(scalar_first)

    weight = torch.nn.Parameter(torch.tensor(PRODUCT_WEIGHT, dtype=torch.float64))
    output = SCALAR_SIDE * (weight @ torch.tensor(PRODUCT_INPUT, dtype=torch.float64))
    ((output - torch.tensor(PRODUCT_TARGET, dtype=torch.float64)) ** 2).sum().backward()
    stepped = weight - LEARNING_RATE * weight.grad

    assert value == pytest.approx(output.tolist(), abs=1e-9)
    assert after == pytest.approx([entry for row in stepped.tolist() for entry in row], abs=1e-9)


def test_scalar_times_vector_product_does_not_depend_on_body_order():
    """The invariant the two above are instances of, stated on its own so a failure names it."""
    first_value, first_after = _scalar_times_vector(True)
    last_value, last_after = _scalar_times_vector(False)

    assert first_value == pytest.approx(last_value, abs=1e-12)
    assert first_after == pytest.approx(last_after, abs=1e-12)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_reported_error_over_a_vector_target_is_the_summed_squared_error(width):
    """The number `validate` reports has to be the function the gradient descends, not a scaled cousin.

    `SquaredDiff.evaluate` used to divide by the component count while `differentiate` did not, so for a
    vector target the reported error was the mean where the gradient was the sum's - off by exactly the
    output width, and invisible at width one. `Crossentropy` and `SoftEntropy` both already summed over
    components, so this also makes the three agree. Nothing on either side of the library could see it: the
    Java suite and this one both passed unchanged when it was fixed, which is why it is pinned here.
    """
    weight = [[0.5 if row == column else 0.25 for column in range(width)] for row in range(width)]
    source = [0.7 - 0.3 * index for index in range(width)]
    target = [0.2 * index - 0.1 for index in range(width)]

    model = Model()
    model += R.source("a")[source].fixed()
    model += (R.out(V.X)["w":width, width] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [Transformation.IDENTITY]
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = weight
    built.load_state_dict(state)

    query = R.out("a")[target] if width > 1 else R.out("a")[target[0]]
    data = built.build_dataset(Dataset([Sample(query, [R.exists("a")])]))
    _, _, reported = built.validate(data)[0]

    output = torch.tensor(weight, dtype=torch.float64) @ torch.tensor(source, dtype=torch.float64)
    squared = (output - torch.tensor(target, dtype=torch.float64)) ** 2

    assert float(reported) == pytest.approx(float(squared.sum()), abs=1e-9)
    if width > 1:
        # and not the mean, which is what it used to be - stated so the test fails in the direction it came from
        assert float(reported) != pytest.approx(float(squared.mean()), abs=1e-9)


#: The reduction cases below train, so they need their own small setup rather than the fixed-weight helpers
REDUCTION_WEIGHT = [[0.5, -0.2], [0.4, -0.6]]
REDUCTION_INPUTS = {"a": [0.5, -0.25], "b": [-0.3, 0.8]}
REDUCTION_TARGETS = {"a": [0.1, 0.2], "b": [-0.4, 0.15]}


def _stepped_under(reduction: str, keys, batch_size: int):
    model = Model()
    for name in keys:
        model += R.source(name)[REDUCTION_INPUTS[name]].fixed()
    model += (R.out(V.X)["w":2, 2] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [Transformation.IDENTITY]

    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(reduction=reduction),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    index = next(i for i, name in state["weight_names"].items() if str(name).strip() == "w")
    state["weights"][index] = REDUCTION_WEIGHT
    built.load_state_dict(state)

    dataset = Dataset([Sample(R.out(k)[REDUCTION_TARGETS[k]], [R.exists(k)]) for k in keys])
    built.train(built.build_dataset(dataset, batch_size=batch_size), epochs=1)
    return [entry for row in built.state_dict()["weights"][index] for entry in row]


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("keys, batch_size", [(["a"], 1), (["a", "b"], 2)], ids=["one-query", "two-queries"])
def test_loss_reduction_steps_like_torch(reduction, keys, batch_size):
    """`reduction` means what it means in torch: it changes the *gradient*, not only the reported number.

    Torch divides by the total element count under `mean` - samples times components, not the batch alone -
    and its gradient follows, which is the whole point of the flag being part of the graph. The engine has
    nowhere to put a tensor, so the same divisor is applied where the batch is known: the minibatch
    accumulation and the single-sample path, which is why both batch sizes are here.

    The rest of this suite pins `reduction="sum"` explicitly because it compares against torch losses written
    with an explicit `.sum()`. This is the one that covers the *default*.
    """
    stepped = _stepped_under(reduction, keys, batch_size)

    weight = torch.nn.Parameter(torch.tensor(REDUCTION_WEIGHT, dtype=torch.float64))
    inputs = torch.tensor([REDUCTION_INPUTS[k] for k in keys], dtype=torch.float64)
    targets = torch.tensor([REDUCTION_TARGETS[k] for k in keys], dtype=torch.float64)
    torch.nn.MSELoss(reduction=reduction)(inputs @ weight.T, targets).backward()
    expected = weight - LEARNING_RATE * weight.grad

    assert stepped == pytest.approx([entry for row in expected.tolist() for entry in row], abs=1e-9)


def test_the_default_reduction_is_torch_s():
    """Stated on its own, so that changing the default fails here rather than somewhere far away."""
    assert MSE().reduction == "mean" == torch.nn.MSELoss().reduction
