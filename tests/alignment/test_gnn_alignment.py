import pytest
import torch
from torch_geometric.nn import GCNConv as TorchGCN, GINConv as TorchGIN, SAGEConv as TorchSAGE

import neuralogic.nn.module as module
from neuralogic.core import Model, R, Settings
from neuralogic.dataset import Dataset, Sample
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import SGD

LEARNING_RATE = 0.1
IN, OUT = 4, 3

#: Three nodes, undirected 0-1 and 1-2, so the degrees differ - 1, 2, 1 - and anything that divides by the
#: wrong one, or counts a node among its own neighbours when it should not, comes out different per node.
EDGES = [(0, 1), (1, 0), (1, 2), (2, 1)]
NODES = 3

FEATURES = [
    [0.7, -0.4, 0.2, 0.9],
    [-0.3, 0.8, -0.6, 0.1],
    [0.5, 0.15, 0.35, -0.75],
]
TARGET = [[1.0, 0.0, -0.5], [0.2, -0.3, 0.8], [-0.6, 0.45, 0.1]]

FIRST = [[0.5, -0.2, 0.1, 0.3], [0.4, -0.6, 0.25, -0.15], [-0.1, 0.2, 0.8, 0.05]]
SECOND = [[0.2, 0.35, -0.4, 0.1], [-0.55, 0.15, 0.6, -0.25], [0.3, -0.45, 0.05, 0.7]]


def _tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def _flat(rows):
    return [entry for row in rows for entry in row]


def _edge_index():
    return torch.tensor(EDGES, dtype=torch.long).t().contiguous()


def _example(edge_value):
    """The graph as facts: a feature vector per node, and an edge fact per direction.

    The edge's *value* matters, and not in a good way - see the note on SAGE below. GCN multiplies by it and
    so wants the 1.0 that leaves a product alone; SAGE and GIN add it, and so want the 0.0 that leaves a sum
    alone. That the same graph has to be spelled two ways for three modules is the finding, not the setup.
    """
    return [R.f(node)[FEATURES[node]] for node in range(NODES)] + [R.e(i, j)[edge_value] for i, j in EDGES]


def _built(gnn_module, weights, edge_value):
    model = Model()
    model += gnn_module
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    for index, value in weights.items():
        state["weights"][index] = value
    built.load_state_dict(state)

    dataset = Dataset([Sample(R.h(node)[TARGET[node]], _example(edge_value)) for node in range(NODES)])
    return built, built.build_dataset(dataset, batch_size=NODES)


def _forward_and_step(built, dataset, indices):
    """One plain SGD step over all three nodes at once, so it is the sum torch's summed loss also takes."""
    value = [[float(v) for v in row] for row in built(dataset)]
    before = {i: built.state_dict()["weights"][i] for i in indices}
    built.train(dataset, epochs=1)
    after = built.state_dict()["weights"]
    return value, {i: _flat(after[i]) for i in indices}, {i: _flat(before[i]) for i in indices}


def _torch_step(layer, forward):
    optimizer = torch.optim.SGD(layer.parameters(), lr=LEARNING_RATE)
    output = forward()
    optimizer.zero_grad()
    ((output - _tensor(TARGET)) ** 2).sum().backward()
    optimizer.step()
    return output


def test_gcn_matches_torch_geometric():
    """GCN normalises by degree and counts a node among its own neighbours; both show up per node.

    The template does that with a self-loop fact and `sqrt` of a count on each side of the edge, where PyG
    does it with `add_self_loops` and a symmetric normalisation - two spellings of one thing, which is why a
    graph with unequal degrees is worth using.
    """
    built, dataset = _built(module.GCNConv(IN, OUT, "h", "f", "e"), {1: FIRST}, edge_value=1.0)
    layer = TorchGCN(IN, OUT, bias=False).double()
    with torch.no_grad():
        layer.lin.weight.copy_(_tensor(FIRST))

    value, after, _ = _forward_and_step(built, dataset, [1])
    expected = _torch_step(
        layer, lambda: layer(_tensor(FEATURES), _edge_index(), torch.ones(len(EDGES), dtype=torch.float64))
    )

    assert _flat(value) == pytest.approx(_flat(expected.tolist()), abs=1e-9)
    assert after[1] == pytest.approx(_flat(layer.lin.weight.tolist()), abs=1e-9)


def test_sage_matches_torch_geometric():
    """SAGE keeps the node and its neighbourhood apart, one weight each, and means over the neighbours.

    **The edges have to carry 0.0 here, and that is a defect rather than a convention.** The rule the module
    emits is `h(I) :- f(J), e(J, I)` with the edge as an ordinary valued body atom, and a rule body combines
    by SUM - so the edge's value is added to every component of the neighbour's feature vector. **Measured**:
    with edges at the natural 1.0, node 0 comes out `[0.36, -0.865, 0.665]` where `W . mean(neighbours)` is
    `[-0.34, -0.765, -0.285]`, the difference being exactly `W . 1`. At 0.0, the additive identity, all nine
    numbers match PyG.

    GCN escapes it only because its rule sets `combination=product`, for which 1.0 is the identity instead.
    Marking the edge atom `hidden` fixes it properly - measured, the value then makes no difference at all,
    7.3 included - and that is the library's own idiom for an atom that is there to ground and not to count.
    """
    built, dataset = _built(module.SAGEConv(IN, OUT, "h", "f", "e"), {0: FIRST, 1: SECOND}, edge_value=0.0)
    layer = TorchSAGE(IN, OUT, bias=False).double()
    with torch.no_grad():
        layer.lin_l.weight.copy_(_tensor(FIRST))     # the neighbourhood
        layer.lin_r.weight.copy_(_tensor(SECOND))    # the node itself

    value, after, _ = _forward_and_step(built, dataset, [0, 1])
    expected = _torch_step(layer, lambda: layer(_tensor(FEATURES), _edge_index()))

    assert _flat(value) == pytest.approx(_flat(expected.tolist()), abs=1e-9)
    assert after[0] == pytest.approx(_flat(layer.lin_l.weight.tolist()), abs=1e-9)
    assert after[1] == pytest.approx(_flat(layer.lin_r.weight.tolist()), abs=1e-9)


def test_gin_matches_torch_geometric():
    """GIN sums the neighbourhood, adds the node, and puts one map over the total.

    The template spells that as two weighted paths rather than one map over a sum, so they only describe the
    same function when the extra map in the self path is the identity - which is what setting the 4x4 weight
    to one does. PyG's eps is 0 by default, so the node counts once.
    """
    identity = [[1.0 if i == j else 0.0 for j in range(IN)] for i in range(IN)]
    built, dataset = _built(module.GINConv(IN, OUT, "h", "f", "e"), {0: FIRST, 1: FIRST, 2: identity}, edge_value=0.0)

    # after constructing the layer, not before: GINConv's own __init__ calls reset_parameters, which resets
    # the module handed to it - so a weight copied in first is thrown away before it is ever used
    layer = TorchGIN(torch.nn.Linear(IN, OUT, bias=False)).double()
    with torch.no_grad():
        layer.nn.weight.copy_(_tensor(FIRST))

    value, after, before = _forward_and_step(built, dataset, [0, 1])
    expected = _torch_step(layer, lambda: layer(_tensor(FEATURES), _edge_index()))

    assert _flat(value) == pytest.approx(_flat(expected.tolist()), abs=1e-9)

    # torch carries one weight where the template carries two, so its step has to be what both of these add
    # up to - which also says the gradient reached each path rather than one of them twice
    both_paths = [
        (start - neighbourhood) + (start - node)
        for start, neighbourhood, node in zip(before[0], after[0], after[1])
    ]
    one_weight = [start - stepped for start, stepped in zip(before[0], _flat(layer.nn.weight.tolist()))]

    assert both_paths == pytest.approx(one_weight, abs=1e-9)
