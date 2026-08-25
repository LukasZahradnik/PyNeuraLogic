import math

import pytest
import torch

from neuralogic.core import Combination, Model, R, Settings, Transformation, V
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.nn.optim import SGD, Adam

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


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("keys, batch_size", [(["a"], 1), (["a", "b"], 2)], ids=["one-query", "two-queries"])
def test_loss_is_what_torch_s_criterion_returns(reduction, keys, batch_size):
    """`loss()` hands back the reduced scalar, which is what a torch criterion call returns.

    Separate from `validate()`, whose per-query values are *not* reduced across the batch - each is summed
    over its own components, torch's `reduction="none"`. Both are useful and they are different quantities;
    the divisor between them is `Result.reductionDivisor`, the same one the trainers scale the gradient by,
    so the reported loss and the descended one cannot drift apart.
    """
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
    data = built.build_dataset(dataset, batch_size=batch_size)

    weight = torch.tensor(REDUCTION_WEIGHT, dtype=torch.float64)
    inputs = torch.tensor([REDUCTION_INPUTS[k] for k in keys], dtype=torch.float64)
    targets = torch.tensor([REDUCTION_TARGETS[k] for k in keys], dtype=torch.float64)
    expected = torch.nn.MSELoss(reduction=reduction)(inputs @ weight.T, targets)

    assert built.loss(data) == pytest.approx(float(expected), abs=1e-9)

    # and the per-query values stay un-reduced, so the two are not accidentally the same call
    per_query = [float(error) for _, _, error in built.validate(data)]
    assert sum(per_query) == pytest.approx(float(((inputs @ weight.T - targets) ** 2).sum()), abs=1e-9)


#: Weight decay and clipping train *two* weights on purpose. A global gradient norm - which is what
#: `clip_grad_norm_` takes - is indistinguishable from a per-weight one until there is more than one weight,
#: and the same goes for a decay applied to the wrong subset. Both feed one head, so a single query already
#: puts a gradient on both.
CLIP_WEIGHTS = {"u": [[0.5, -0.2], [0.4, -0.6]], "v": [[0.3, 0.7], [-0.1, 0.25]]}
CLIP_SOURCES = {"a": ([0.5, -0.25], [-0.7, 0.4]), "b": ([-0.3, 0.8], [0.6, 0.1])}
CLIP_TARGETS = {"a": [0.1, 0.2], "b": [-0.4, 0.15]}


def _clip_model(keys, optimizer, reduction="sum", **settings):
    model = Model()
    for name in keys:
        first, second = CLIP_SOURCES[name]
        model += R.first(name)[first].fixed()
        model += R.second(name)[second].fixed()
    model += (R.out(V.X)["u":2, 2] <= R.first(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += (R.out(V.X)["v":2, 2] <= R.second(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [Transformation.IDENTITY]

    built = model.build(
        Settings(
            optimizer=optimizer,
            error_function=MSE(reduction=reduction),
            iso_value_compression=False,
            chain_pruning=False,
            **settings,
        )
    )
    state = built.state_dict()
    indices = {
        name: next(i for i, weight in state["weight_names"].items() if str(weight).strip() == name)
        for name in CLIP_WEIGHTS
    }
    for name, index in indices.items():
        state["weights"][index] = CLIP_WEIGHTS[name]
    built.load_state_dict(state)
    return built, indices


def _lrnn_stepped(keys, optimizer, batch_size=1, single_sample=False, reduction="sum", **settings):
    built, indices = _clip_model(keys, optimizer, reduction, **settings)
    dataset = Dataset([Sample(R.out(k)[CLIP_TARGETS[k]], [R.exists(k)]) for k in keys])
    data = built.build_dataset(dataset, batch_size=batch_size)

    built.train(data._samples[0] if single_sample else data, epochs=1)

    state = built.state_dict()
    return [entry for name in CLIP_WEIGHTS for row in state["weights"][indices[name]] for entry in row]


def _torch_stepped(keys, make_optimizer, clip_norm=None, clip_value=None):
    weights = {
        name: torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        for name, value in CLIP_WEIGHTS.items()
    }
    optimizer = make_optimizer(list(weights.values()))

    loss = torch.zeros((), dtype=torch.float64)
    for key in keys:
        first, second = (torch.tensor(source, dtype=torch.float64) for source in CLIP_SOURCES[key])
        output = weights["u"] @ first + weights["v"] @ second
        loss = loss + (output - torch.tensor(CLIP_TARGETS[key], dtype=torch.float64)).pow(2).sum()
    loss.backward()

    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(list(weights.values()), clip_norm)
    if clip_value is not None:
        torch.nn.utils.clip_grad_value_(list(weights.values()), clip_value)
    optimizer.step()

    return [entry for name in CLIP_WEIGHTS for row in weights[name].detach().tolist() for entry in row]


@pytest.mark.parametrize("keys, batch_size", [(["a"], 1), (["a", "b"], 2)], ids=["one-query", "two-queries"])
def test_sgd_weight_decay_is_torch_s(keys, batch_size):
    """`weight_decay` is torch's: the penalty is added to the gradient, so the learning rate scales it too.

    Both trainer paths, because the decay is applied inside the optimizer while the reduction is applied
    outside it, and only a batch above one exercises the accumulation between them.
    """
    stepped = _lrnn_stepped(keys, SGD(lr=LEARNING_RATE, weight_decay=0.3), batch_size)
    expected = _torch_stepped(keys, lambda p: torch.optim.SGD(p, lr=LEARNING_RATE, weight_decay=0.3))

    assert stepped == pytest.approx(expected, abs=1e-12)
    # and it is not a no-op, so a decay that never reached the optimizer would fail here
    assert stepped != pytest.approx(_lrnn_stepped(keys, SGD(lr=LEARNING_RATE), batch_size), abs=1e-12)


def test_adam_weight_decay_is_coupled_l2_and_not_adamw():
    """Which of torch's two decays this is, stated so that changing it fails here.

    `Adam(weight_decay=)` adds the penalty to the gradient *before* the moments, so it accumulates in them
    and the adaptive scaling applies to it. `AdamW` applies it to the weight directly, bypassing that. They
    are different updates from the same numbers, and this is the first.
    """
    stepped = _lrnn_stepped(["a"], Adam(lr=LEARNING_RATE, weight_decay=0.3))

    coupled = _torch_stepped(["a"], lambda p: torch.optim.Adam(p, lr=LEARNING_RATE, weight_decay=0.3))
    decoupled = _torch_stepped(["a"], lambda p: torch.optim.AdamW(p, lr=LEARNING_RATE, weight_decay=0.3))

    assert stepped == pytest.approx(coupled, abs=1e-12)
    assert stepped != pytest.approx(decoupled, abs=1e-12)


@pytest.mark.parametrize("keys, batch_size", [(["a"], 1), (["a", "b"], 2)], ids=["one-query", "two-queries"])
def test_clip_grad_norm_is_torch_s(keys, batch_size):
    """One norm over all the weights together, and torch's exact factor - including its epsilon.

    `clip_grad_norm_` divides by `total_norm + 1e-6` rather than by `total_norm`, so a clipped gradient
    comes out a shade under the bound instead of exactly at it. The tolerance here is tight enough that
    dropping the epsilon fails.
    """
    stepped = _lrnn_stepped(keys, SGD(lr=LEARNING_RATE), batch_size, clip_grad_norm=0.05)
    expected = _torch_stepped(keys, lambda p: torch.optim.SGD(p, lr=LEARNING_RATE), clip_norm=0.05)

    assert stepped == pytest.approx(expected, abs=1e-12)


def test_clip_grad_norm_is_global_and_not_per_weight():
    """The distinguishing case: clipping each weight to the bound separately is a different step.

    Nothing above separates the two, since torch is the reference for both this and the per-weight reading;
    this states the difference in the engine's own terms.
    """
    together = _lrnn_stepped(["a"], SGD(lr=LEARNING_RATE), clip_grad_norm=0.05)

    weights = {
        name: torch.nn.Parameter(torch.tensor(v, dtype=torch.float64)) for name, v in CLIP_WEIGHTS.items()
    }
    first, second = (torch.tensor(source, dtype=torch.float64) for source in CLIP_SOURCES["a"])
    output = weights["u"] @ first + weights["v"] @ second
    (output - torch.tensor(CLIP_TARGETS["a"], dtype=torch.float64)).pow(2).sum().backward()
    optimizer = torch.optim.SGD(list(weights.values()), lr=LEARNING_RATE)
    for weight in weights.values():  # each on its own, which is the reading this must not be
        torch.nn.utils.clip_grad_norm_([weight], 0.05)
    optimizer.step()
    separately = [e for name in CLIP_WEIGHTS for row in weights[name].detach().tolist() for e in row]

    assert together != pytest.approx(separately, abs=1e-12)


def test_a_gradient_under_the_clip_norm_is_left_exactly_alone():
    """Torch clamps the factor at one, so an unclipped step must be bit-identical to no clipping at all."""
    unclipped = _lrnn_stepped(["a"], SGD(lr=LEARNING_RATE))
    generous = _lrnn_stepped(["a"], SGD(lr=LEARNING_RATE), clip_grad_norm=1e6)

    assert generous == unclipped


@pytest.mark.parametrize("keys, batch_size", [(["a"], 1), (["a", "b"], 2)], ids=["one-query", "two-queries"])
def test_clip_grad_value_is_torch_s(keys, batch_size):
    """`clip_grad_value_`: every element clamped to +-the bound, each weight on its own."""
    stepped = _lrnn_stepped(keys, SGD(lr=LEARNING_RATE), batch_size, clip_grad_value=0.05)
    expected = _torch_stepped(keys, lambda p: torch.optim.SGD(p, lr=LEARNING_RATE), clip_value=0.05)

    assert stepped == pytest.approx(expected, abs=1e-12)


def test_weight_decay_is_not_clipped():
    """Torch's order: `clip_grad_norm_` runs on the gradient, then `step()` adds the decay to it.

    So the decay term survives a bound that the gradient itself is crushed to - which is the whole reason
    the two live in different places here as well, clipping in the trainer and decay in the optimizer.
    """
    stepped = _lrnn_stepped(["a"], SGD(lr=LEARNING_RATE, weight_decay=0.3), clip_grad_norm=1e-4)
    expected = _torch_stepped(
        ["a"], lambda p: torch.optim.SGD(p, lr=LEARNING_RATE, weight_decay=0.3), clip_norm=1e-4
    )

    assert stepped == pytest.approx(expected, abs=1e-12)
    # a decay that had been clipped along with the gradient would be all but gone, leaving the weights put
    assert stepped != pytest.approx(_lrnn_stepped(["a"], SGD(lr=LEARNING_RATE), clip_grad_norm=1e-4), abs=1e-9)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_one_sample_trains_the_same_whether_or_not_it_is_in_a_collection(reduction):
    """`train(sample)` and `train([sample])` are the same descent, and were not.

    The single-sample path reimplements the trainer's rather than calling it, so it stood outside the
    reduction entirely: under `mean` it stepped as if under `sum`, off by the target width. Nothing in the
    reported error showed it, because the reporting was shared and only the gradient diverged.
    """
    listed = _lrnn_stepped(["a"], SGD(lr=LEARNING_RATE), reduction=reduction)
    single = _lrnn_stepped(["a"], SGD(lr=LEARNING_RATE), reduction=reduction, single_sample=True)

    assert single == listed
    if reduction == "mean":
        # and under `mean` that is genuinely a different step from `sum`, which is what it used to take
        assert listed != pytest.approx(_lrnn_stepped(["a"], SGD(lr=LEARNING_RATE), reduction="sum"), abs=1e-12)


def test_clipping_reaches_the_single_sample_path_too():
    """The same seam, for clipping: it is applied next to the reduction, so it is missed the same way."""
    clipped = _lrnn_stepped(["a"], SGD(lr=LEARNING_RATE), 1, single_sample=True, clip_grad_norm=0.05)
    expected = _torch_stepped(["a"], lambda p: torch.optim.SGD(p, lr=LEARNING_RATE), clip_norm=0.05)

    assert clipped == pytest.approx(expected, abs=1e-12)


#: The activation cases above run at a *positive* pre-activation (`START * INPUT` is `0.63`), so a leaky
#: slope never engages there at all. These drive it negative, which is the only place the slope exists.
NEGATIVE_START = -0.9


def _leaky_value_and_gradient(transformation):
    """The same one-weight model as `_built`, but with the pre-activation on the negative side."""
    model = Model()
    model += R.source("a")[INPUT].fixed()
    model += (R.out(V.X)["w":1, 1] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.out / 1 | [transformation]
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
    state["weights"][index] = NEGATIVE_START
    built.load_state_dict(state)

    dataset = built.build_dataset(Dataset([Sample(R.out("a")[TARGET], [R.exists("a")])]))
    value = float(built(dataset)[0])
    before = float(built.state_dict()["weights"][index])
    built.train(dataset, epochs=1)
    gradient = (before - float(built.state_dict()["weights"][index])) / LEARNING_RATE
    return value, gradient


def _torch_leaky_value_and_gradient(slope):
    weight = torch.nn.Parameter(torch.tensor([NEGATIVE_START], dtype=torch.float64))
    output = torch.nn.functional.leaky_relu(weight * INPUT, slope)
    ((output - torch.tensor([TARGET], dtype=torch.float64)) ** 2).sum().backward()
    return float(output.item()), float(weight.grad.item())


@pytest.mark.parametrize("slope", [0.2, 0.5, 0.0, 1.0], ids=["pyg-default", "half", "hard-relu", "linear"])
def test_leaky_relu_takes_the_slope_it_is_given(slope):
    """The slope used to be one mutable static, global to the JVM, so no rule could ask for one.

    Both halves are checked, and the gradient is the one that matters: on the negative side the slope *is*
    the derivative, so a value that agreed while the gradient did not would be the worse failure.
    """
    value, gradient = _leaky_value_and_gradient(Transformation.LEAKY_RELU(slope))
    expected_value, expected_gradient = _torch_leaky_value_and_gradient(slope)

    assert value == pytest.approx(expected_value, abs=1e-12)
    assert gradient == pytest.approx(expected_gradient, abs=1e-12)


def test_leaky_relu_without_a_slope_is_still_the_backend_default():
    """Asking for nothing has to keep meaning what it meant, which is `0.01`."""
    value, gradient = _leaky_value_and_gradient(Transformation.LEAKY_RELU)
    expected_value, expected_gradient = _torch_leaky_value_and_gradient(0.01)

    assert value == pytest.approx(expected_value, abs=1e-12)
    assert gradient == pytest.approx(expected_gradient, abs=1e-12)


def test_two_rules_in_one_template_can_use_different_slopes():
    """What a single static could not express, and the reason this is per rule at all."""
    model = Model()
    model += R.source("a")[-4.0].fixed()
    model += (R.soft(V.X)["u":1, 1] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.soft / 1 | [Transformation.LEAKY_RELU(0.5)]
    model += (R.hard(V.X)["v":1, 1] <= R.source(V.X)) | [Combination.SUM, Transformation.IDENTITY]
    model += R.hard / 1 | [Transformation.LEAKY_RELU(0.01)]

    built = model.build(
        Settings(
            optimizer=SGD(lr=0.0),
            error_function=MSE(reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    for name in ("u", "v"):
        index = next(i for i, n in state["weight_names"].items() if str(n).strip() == name)
        state["weights"][index] = 1.0
    built.load_state_dict(state)

    dataset = built.build_dataset(
        Dataset([
            Sample(R.soft("a")[0.0], [R.exists("a")]),
            Sample(R.hard("a")[0.0], [R.exists("a")]),
        ])
    )

    assert [float(out) for out in built(dataset)] == pytest.approx([-2.0, -0.04], abs=1e-12)


#: The units of a queried atom used to depend on how many rules defined it, so these build the same
#: prediction out of one, two and three rules and ask for the same answer each time.
UNITS_FEATURES = [0.4, -0.9]
UNITS_WEIGHT = 1.5


def _queried_value(rule_count, error_function, stated=None):
    """out(X) defined by `rule_count` rules, each from its own source, so only the count varies."""
    model = Model()
    for i in range(rule_count):
        model += R.src(f"s{i}", V.X)[UNITS_FEATURES[i % len(UNITS_FEATURES)]].fixed()
        model += (R.out(V.X)[f"w{i}":1, 1] <= R.src(f"s{i}", V.X)) | [Combination.SUM, Transformation.IDENTITY]
    if stated is not None:
        model += R.out / 1 | [stated]

    built = model.build(
        Settings(
            optimizer=SGD(lr=0.0),
            error_function=error_function,
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    for index in state["weights"]:
        if index >= 0:
            state["weights"][index] = UNITS_WEIGHT
    built.load_state_dict(state)

    dataset = built.build_dataset(Dataset([Sample(R.out("a")[1.0], [R.exists("a")])]))
    return float(built(dataset)[0])


def _raw_sum(rule_count):
    return sum(UNITS_WEIGHT * UNITS_FEATURES[i % len(UNITS_FEATURES)] for i in range(rule_count))


@pytest.mark.parametrize("rule_count", [1, 2, 3])
def test_an_inferred_output_function_does_not_depend_on_the_rule_count(rule_count):
    """`CrossEntropy(with_logits=False)` wants a probability, whatever the shape of the template.

    It used to get one only when the queried predicate had exactly *one* rule. A queried atom whose template
    says IDENTITY carries a plain `Combination.State`, and that refused to take a transformation at all -
    reading "there is no transformation here" as "one cannot be put here". With a single rule
    `StateInitializer` had already turned the state into a `Transformation.State`, which does take one, so
    the inference landed; with two or more it was silently dropped and `test()` handed back a logit.

    Silent in both directions, which is what makes it worth a test: a caller that squashes again maps the
    whole unit interval into [0.5, 0.731], and one that does not read logits as probabilities.
    """
    value = _queried_value(rule_count, CrossEntropy(with_logits=False))
    expected = 1 / (1 + math.exp(-_raw_sum(rule_count)))

    assert value == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("rule_count", [1, 2, 3])
def test_a_stated_output_function_is_still_left_alone(rule_count):
    """The other half, and the one that was already right: inference must not overrule the template.

    Stating IDENTITY is how a head says it is already the final quantity. If the fix above had been "always
    squash the query", this is what would have broken.
    """
    value = _queried_value(rule_count, CrossEntropy(with_logits=False), stated=Transformation.IDENTITY)

    assert value == pytest.approx(_raw_sum(rule_count), abs=1e-12)


@pytest.mark.parametrize("rule_count", [1, 2, 3])
def test_with_logits_infers_nothing_whatever_the_rule_count(rule_count):
    """`with_logits=True` says the head is a logit, so nothing should be put on top of it."""
    value = _queried_value(rule_count, CrossEntropy(with_logits=True))

    assert value == pytest.approx(_raw_sum(rule_count), abs=1e-12)


@pytest.mark.parametrize("rule_count", [1, 2, 3])
def test_the_inferred_output_function_reaches_the_gradient_too(rule_count):
    """The other half, and the one the value alone would not have caught.

    The units of the *reported* number were what made this visible, but the transformation the inference
    installs is on the queried neuron, so it is in the backward pass as well: with it dropped, the step at two
    and three rules matched a hand-computed BCE through neither sigmoid(sum) nor sum.

    Oracle: with p = sigmoid(z) and z = sum(w_i x_i), binary cross-entropy has dL/dz = p - t, so one SGD step
    is w_i - lr (p - t) x_i. This project has twice had to defend the same invariant - that the number
    reported is the function being descended - so the gradient gets its own assertion rather than being
    assumed to follow the value.
    """
    learning_rate = 0.1
    model = Model()
    for i in range(rule_count):
        model += R.src(f"s{i}", V.X)[UNITS_FEATURES[i % len(UNITS_FEATURES)]].fixed()
        model += (R.out(V.X)[f"w{i}":1, 1] <= R.src(f"s{i}", V.X)) | [Combination.SUM, Transformation.IDENTITY]

    built = model.build(
        Settings(
            optimizer=SGD(lr=learning_rate),
            error_function=CrossEntropy(with_logits=False, reduction="sum"),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    indices = sorted(i for i in state["weights"] if i >= 0)
    for index in indices:
        state["weights"][index] = UNITS_WEIGHT
    built.load_state_dict(state)

    built.train(built.build_dataset(Dataset([Sample(R.out("a")[1.0], [R.exists("a")])])), epochs=1)
    stepped = [float(built.state_dict()["weights"][index]) for index in indices]

    features = [UNITS_FEATURES[i % len(UNITS_FEATURES)] for i in range(rule_count)]
    probability = 1 / (1 + math.exp(-_raw_sum(rule_count)))
    expected = [UNITS_WEIGHT - learning_rate * (probability - 1.0) * x for x in features]

    assert stepped == pytest.approx(expected, abs=1e-12)
