from typing import List
import itertools

import pytest

from neuralogic import manual_seed
from neuralogic.nn import get_evaluator
from neuralogic.core import Settings, R, V, Template, Transformation
from neuralogic.dataset import Dataset, Sample
from neuralogic.optim import SGD


@pytest.mark.parametrize(
    "n,expected",
    [
        (2, [0, 1, 1, 0]),  # Number of inputs and expected output
        (3, [0, 1, 1, 0, 1, 0, 0, 1]),
        (4, [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
        (5, [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
    ],
)
def test_xor_generalization_accurate(n: int, expected: List[int]) -> None:
    manual_seed(0)
    max_number_of_max_vars = 20

    dataset = Dataset()
    template = Template()

    template += R.xor_at(0) <= R.val_at(0)
    template += R.xor_at(V.Y)["a":1, 8] <= (R.val_at(V.Y)["b":8, 1], R.xor_at(V.X)["c":8, 1], R.special.next(V.X, V.Y))

    dataset.add_samples(
        [
            Sample(R.xor_at(1)[0], [R.val_at(0)[0], R.val_at(1)[0]]),
            Sample(R.xor_at(1)[1], [R.val_at(0)[0], R.val_at(1)[1]]),
            Sample(R.xor_at(1)[1], [R.val_at(0)[1], R.val_at(1)[0]]),
            Sample(R.xor_at(1)[0], [R.val_at(0)[1], R.val_at(1)[1]]),
        ]
    )

    settings = Settings(
        epochs=5000, rule_transformation=Transformation.TANH, relation_transformation=Transformation.IDENTITY
    )

    evaluator = get_evaluator(template, settings)
    evaluator.train(dataset, generator=False)

    # build the dataset for n inputs
    products = itertools.product([0, 1], repeat=n)
    n_dataset = Dataset()

    for example in products:
        n_dataset.add_sample(Sample(R.xor_at(n - 1)[0], [R.val_at(i)[int(val)] for i, val in enumerate(example)]))

    for expected_value, predicted in zip(expected, evaluator.test(n_dataset)):
        assert expected_value == predicted


@pytest.mark.parametrize(
    "n,expected",
    [
        (2, [0, 1, 1, 0]),  # Number of inputs and expected output
        (3, [0, 1, 1, 0, 1, 0, 0, 1]),
        (4, [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
    ],
)
def test_xor_generalization(n: int, expected: List[int]) -> None:
    """Tests xor generalization"""
    # fmt: off
    manual_seed(0)
    template = Template()

    # We have three weights in total named "a", "b" and "c"
    template.add_rules([

        # This rule does xor for the last pair
        R.xor(V.X, V.Y)["a":1, 8] <= (
            R.x(V.X)["b":8, 1], R.x(V.Y)["c":8, 1], R.hidden.xy(V.X, V.Y), R.hidden.n(V.Y)
        ),

        # This rule recursively evaluates xor for X and xor(Y, Z)
        R.xor(V.X, V.Y)["a":1, 8] <= (
            R.x(V.X)["b":8, 1], R.xor(V.Y, V.Z)["c":8, 1], R.hidden.xy(V.X, V.Y), R.hidden.xy(V.Y, V.Z)
        ),

        # Helper rule so that queries are just R.xor
        (R.xor <= R.xor(0, V.X))
    ])

    # The training dataset to train xor on two inputs x(0) and x(1), n(1) is means the max index of input is 1
    # x(0, 1) defines which input should be "xor-ed" together
    dataset = Dataset()
    dataset.add_samples([
        Sample(R.xor[0.0], [R.xy(0, 1), R.x(0)[0.0], R.x(1)[0.0], R.n(1)]),
        Sample(R.xor[1.0], [R.xy(0, 1), R.x(0)[1.0], R.x(1)[0.0], R.n(1)]),
        Sample(R.xor[1.0], [R.xy(0, 1), R.x(0)[0.0], R.x(1)[1.0], R.n(1)]),
        Sample(R.xor[0.0], [R.xy(0, 1), R.x(0)[1.0], R.x(1)[1.0], R.n(1)]),
    ])

    settings = Settings(optimizer=SGD(), epochs=300)
    neuralogic_evaluator = get_evaluator(template, settings)

    # Train on the dataset with two var input
    neuralogic_evaluator.train(dataset, generator=False)

    # Get all products of lenght of n (all inputs of n vars)
    products = itertools.product([0.0, 1.0], repeat=n)

    n_dataset = Dataset()

    for example in products:
        # We make connections xy(0, 1), xy(1, 2), .. xy(n - 2, n - 1) and add them into the example
        fact_example = [R.xy(i, i + 1) for i in range(n - 1)]

        # Define values of individual variables (e.g., 0, 1, 0 -> R.x(0)[0], R.x(1)[1], R.x(2)[0]
        fact_example.extend([R.x(i)[v] for i, v in enumerate(example)])

        # Add info about the maximum index
        fact_example.append(R.n(n - 1))

        # Add example and query to the dataset, the query has some default value (1.0) as we do not care about the label
        n_dataset.add(R.xor, fact_example)

    # Check that we predicted correct values for n inputs for model trained on 2 inputs
    for expected_value, predicted in zip(expected, neuralogic_evaluator.test(n_dataset)):
        assert expected_value == round(predicted)
