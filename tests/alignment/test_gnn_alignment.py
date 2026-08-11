import pytest
import torch
from torch_geometric.nn import (
    GCNConv as TorchGCN,
    GINConv as TorchGIN,
    SAGEConv as TorchSAGE,
    SGConv as TorchSG,
    TAGConv as TorchTAG,
)

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
THIRD = [[0.15, -0.3, 0.45, -0.6], [0.7, 0.05, -0.2, 0.35], [-0.4, 0.6, 0.1, -0.25]]


def _tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def _flat(rows):
    return [entry for row in rows for entry in row]


def _edge_index():
    return torch.tensor(EDGES, dtype=torch.long).t().contiguous()


def _example(edge_value):
    """The graph as facts: a feature vector per node, and an edge fact per direction.

    The natural 1.0 throughout. It did not use to be safe: a module passing the edge as an ordinary valued
    body atom had its value added to every component of the neighbour's features, since a rule body combines
    by SUM - so the same graph needed spelling two ways depending on the module. The edge atoms are `hidden`
    now, which is what an atom there to ground rather than to count should be, and the value is ignored.
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

    This is the case that found the edge-value defect: the rule is `h(I) :- f(J), e(J, I)` and a body combines
    by SUM, so while the edge was an ordinary valued atom its value was added to every component of the
    neighbour's features. **Measured** at the time, node 0 came out `[0.36, -0.865, 0.665]` where
    `W . mean(neighbours)` is `[-0.34, -0.765, -0.285]` - the difference being exactly `W . 1`. GCN never
    showed it because its rule sets `combination=product`, where 1.0 is the identity instead.
    """
    built, dataset = _built(module.SAGEConv(IN, OUT, "h", "f", "e"), {0: FIRST, 1: SECOND}, edge_value=1.0)
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
    built, dataset = _built(module.GINConv(IN, OUT, "h", "f", "e"), {0: FIRST, 1: FIRST, 2: identity}, edge_value=1.0)

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

@pytest.mark.parametrize("k", [1, 2, 3])
def test_sg_matches_torch_geometric(k):
    """SGConv walks k hops and puts one weight over the total, normalised at every hop.

    Two hops is the case worth having. It disagreed for a long time and looked like a defect in composing a
    body where one derived atom appears at two bindings - the normalisation atom does, once per hop. It was
    not: all three node queries sat in one dataset and the forward pass invalidated every sample before
    evaluating any of them, so the ones sharing an intermediate normalisation atom compounded onto the
    first. At one hop nothing is shared, which is why one hop always looked fine.
    """
    built, dataset = _built(module.SGConv(IN, OUT, "h", "f", "e", k=k), {1: FIRST}, edge_value=1.0)
    layer = TorchSG(IN, OUT, K=k, bias=False).double()
    with torch.no_grad():
        layer.lin.weight.copy_(_tensor(FIRST))

    value, after, _ = _forward_and_step(built, dataset, [1])
    expected = _torch_step(
        layer, lambda: layer(_tensor(FEATURES), _edge_index(), torch.ones(len(EDGES), dtype=torch.float64))
    )

    assert _flat(value) == pytest.approx(_flat(expected.tolist()), abs=1e-9)
    assert after[1] == pytest.approx(_flat(layer.lin.weight.tolist()), abs=1e-9)


def test_tag_matches_torch_geometric():
    """TAGConv sums every hop up to k, one weight per hop, and normalises *without* self-loops.

    The missing self-loops are the point of testing it next to SGConv: PyG calls `gcn_norm` with
    `add_self_loops=False` here and `True` there, so the same graph divides by a different degree in the two
    modules - 1, 2, 1 rather than 2, 3, 2 - and a template that copied GCN's normalisation wholesale would
    pass at one hop on a regular graph and fail here.
    """
    weights = {0: FIRST, 1: SECOND, 2: THIRD}
    built, dataset = _built(module.TAGConv(IN, OUT, "h", "f", "e", k=2), weights, edge_value=1.0)
    layer = TorchTAG(IN, OUT, K=2, bias=False).double()
    with torch.no_grad():
        for index, weight in weights.items():
            layer.lins[index].weight.copy_(_tensor(weight))

    value, after, _ = _forward_and_step(built, dataset, [0, 1, 2])
    expected = _torch_step(
        layer, lambda: layer(_tensor(FEATURES), _edge_index(), torch.ones(len(EDGES), dtype=torch.float64))
    )

    assert _flat(value) == pytest.approx(_flat(expected.tolist()), abs=1e-9)
    for index in weights:
        assert after[index] == pytest.approx(_flat(layer.lins[index].weight.tolist()), abs=1e-9)


def test_queries_of_one_example_do_not_leak_into_each_other():
    """Every node of one graph asked in a single forward call, which is what node classification is.

    Kept separate from the tests above because they build a fresh example object per node, so their three
    networks share nothing - and that is exactly the shape that hid the leak. Here one example carries all
    three queries, so they share the intermediate normalisation atoms, and a forward pass that invalidated
    them once before evaluating anything would give the first node its right answer and compound the rest.
    """
    model = Model()
    model += module.SGConv(IN, OUT, "h", "f", "e", k=2)
    built = model.build(
        Settings(
            optimizer=SGD(lr=LEARNING_RATE),
            error_function=MSE(),
            iso_value_compression=False,
            chain_pruning=False,
        )
    )
    state = built.state_dict()
    state["weights"][1] = FIRST
    built.load_state_dict(state)

    one = _example(1.0)  # one object, so one example carrying three queries
    together = built.build_dataset(
        Dataset([Sample(R.h(node)[TARGET[node]], one) for node in range(NODES)]), batch_size=NODES
    )

    at_once = [[float(v) for v in row] for row in built(together)]
    one_by_one = [[float(v) for v in built(together[node])[0]] for node in range(NODES)]

    assert _flat(at_once) == pytest.approx(_flat(one_by_one), abs=1e-12)
