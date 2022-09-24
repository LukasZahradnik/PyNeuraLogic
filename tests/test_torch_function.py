import torch
from torch.nn import Sequential

import neuralogic
from neuralogic.core import Relation, Template, R
from neuralogic.nn.torch_function import NeuraLogic
import neuralogic.nn.functional as F


def test_torch_function_with_parameters():
    torch.manual_seed(1)
    neuralogic.manual_seed(1)

    template = Template()
    template += (Relation.xor[1, 8] <= Relation.xy) | [F.identity]
    template += Relation.xor / 0 | [F.identity]

    def to_logic(tensor_data):
        return [Relation.xy[tensor_data]]

    torch_train_set = [
        (torch.tensor([0.0, 0.0]), torch.tensor(0.0)),
        (torch.tensor([0.0, 1.0]), torch.tensor(1.0)),
        (torch.tensor([1.0, 0.0]), torch.tensor(1.0)),
        (torch.tensor([1.0, 1.0]), torch.tensor(0.0)),
    ]

    sequential = Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        NeuraLogic(
            template,
            [
                R.xy[
                    8,
                ]
            ],
            R.xor,
            to_logic,
        ),
        torch.nn.Sigmoid(),
    )

    optimizer = torch.optim.SGD(sequential.parameters(), lr=0.1)
    loss = torch.nn.MSELoss()

    for _ in range(300):
        for input_data, label in torch_train_set:
            output = sequential(input_data)
            loss_value = loss(output, label)

            optimizer.zero_grad(set_to_none=True)
            loss_value.backward()
            optimizer.step()

    for input_data, label in torch_train_set:
        out = int(round(float(sequential(input_data))))
        float_label = int(label)

        assert out == float_label


def test_torch_function_without_parameters():
    torch.manual_seed(1)
    neuralogic.manual_seed(1)

    template = Template()
    template += (Relation.xor <= Relation.xy) | [F.identity]
    template += Relation.xor / 0 | [F.identity]

    def to_logic(tensor_data):
        return [Relation.xy[tensor_data]]

    torch_train_set = [
        (torch.tensor([0.0, 0.0]), torch.tensor(0.0)),
        (torch.tensor([0.0, 1.0]), torch.tensor(1.0)),
        (torch.tensor([1.0, 0.0]), torch.tensor(1.0)),
        (torch.tensor([1.0, 1.0]), torch.tensor(0.0)),
    ]

    sequential = Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.Tanh(),
        NeuraLogic(
            template,
            [
                R.xy[
                    8,
                ]
            ],
            R.xor,
            to_logic,
        ),
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid(),
    )

    optimizer = torch.optim.SGD(sequential.parameters(), lr=0.1)
    loss = torch.nn.MSELoss()

    for _ in range(400):
        for input_data, label in torch_train_set:
            output = sequential(input_data)
            loss_value = loss(output, label)

            optimizer.zero_grad(set_to_none=True)
            loss_value.backward()
            optimizer.step()

    for input_data, label in torch_train_set:
        out = int(round(float(sequential(input_data))))
        float_label = int(label)

        assert out == float_label
