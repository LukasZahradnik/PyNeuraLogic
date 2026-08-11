from neuralogic.core import Model, Transformation, Aggregation
from neuralogic.nn.module import (
    RGCNConv,
    SAGEConv,
    GCNConv,
    GINConv,
    TAGConv,
    GATv2Conv,
    SGConv,
    APPNPConv,
    ResGatedGraphConv,
)


def test_rgcnconv():
    model = Model()

    model += RGCNConv(1, 2, "h1", "h0", "_edge", ["a", "b", "c"])
    model_str = str(model).split("\n")

    assert model_str[0] == "h1(I) :- {2, 1} h0(I). [aggregation=avg]"
    assert model_str[1] == "h1(I) :- {2, 1} h0(J), *edge(J, a, I). [aggregation=avg]"
    assert model_str[2] == "h1(I) :- {2, 1} h0(J), *edge(J, b, I). [aggregation=avg]"
    assert model_str[3] == "h1(I) :- {2, 1} h0(J), *edge(J, c, I). [aggregation=avg]"
    assert model_str[4] == "h1/1 [transformation=identity]"


def test_rgcnconv_relations_edge_replace():
    model = Model()

    model += RGCNConv(1, 2, "h1", "h0", None, ["a", "b", "c"], Transformation.SIGMOID)
    model_str = str(model).split("\n")

    assert model_str[0] == "h1(I) :- {2, 1} h0(I). [aggregation=avg]"
    assert model_str[1] == "h1(I) :- {2, 1} h0(J), a(J, I). [aggregation=avg]"
    assert model_str[2] == "h1(I) :- {2, 1} h0(J), b(J, I). [aggregation=avg]"
    assert model_str[3] == "h1(I) :- {2, 1} h0(J), c(J, I). [aggregation=avg]"
    assert model_str[4] == "h1/1 [transformation=sigmoid]"


def test_gcnconv():
    model = Model()

    model += GCNConv(1, 2, "h1", "h0", "edge")
    model_str = str(model).split("\n")

    assert model_str[0] == "<1.0> h1__edge(I, I)."
    assert model_str[1] == "h1__edge(I, J) :- edge(I, J)."
    assert model_str[2] == "h1__edge_count(I, J) :- h1__edge(X, J). [aggregation=sum]"
    assert model_str[3] == "h1__edge_count(I, J) :- h1__edge(X, I). [aggregation=sum]"
    assert model_str[4] == "h1__edge_count/2 [transformation=inverse, combination=product]"
    assert (
        model_str[5]
        == "{2, 1} h1(I) :- h0(J), h1__edge(J, I), sqrt(h1__edge_count(J, I)). [combination=product, aggregation=sum]"
    )


def test_sageconv():
    model = Model()

    model += SAGEConv(1, 2, "h1", "h0", "_edge")
    model_str = str(model).split("\n")

    assert model_str[0] == "{2, 1} h1(I) :- h0(J), *edge(J, I). [aggregation=avg]"
    assert model_str[1] == "{2, 1} h1(I) :- h0(I). [aggregation=avg]"


def test_tagconv():
    """TAGConv normalises by the plain degree - no self-loops, which is what PyG's TAGConv does."""
    counting = [
        "h1__edge_count(I, J) :- *edge(X, J). [aggregation=sum]",
        "h1__edge_count(I, J) :- *edge(X, I). [aggregation=sum]",
        "h1__edge_count/2 [transformation=inverse, combination=product]",
    ]
    zero_hop = "{2, 1} h1(I0) :- h0(I0). [combination=product, aggregation=sum]"
    one_hop = (
        "{2, 1} h1(I0) :- h0(I1), *edge(I1, I0), sqrt(h1__edge_count(I1, I0)). "
        "[combination=product, aggregation=sum]"
    )
    two_hop = (
        "{2, 1} h1(I0) :- h0(I2), *edge(I1, I0), sqrt(h1__edge_count(I1, I0)), *edge(I2, I1), "
        "sqrt(h1__edge_count(I2, I1)). [combination=product, aggregation=sum]"
    )
    identity = "h1/1 [transformation=identity]"

    model = Model()
    model += TAGConv(1, 2, "h1", "h0", "_edge")

    assert str(model).split("\n")[:7] == [*counting, zero_hop, one_hop, two_hop, identity]

    model = Model()
    model += TAGConv(1, 2, "h1", "h0", "_edge", 1)

    assert str(model).split("\n")[:6] == [*counting, zero_hop, one_hop, identity]

    model = Model()
    model += TAGConv(1, 2, "h1", "h0", "_edge", 1, normalize=False)

    assert str(model).split("\n")[:3] == [
        zero_hop,
        "{2, 1} h1(I0) :- h0(I1), *edge(I1, I0). [combination=product, aggregation=sum]",
        identity,
    ]


def test_gatv2conv():
    """The attention is one number per edge, softmaxed over the neighbours, and the edge only grounds.

    `agg_terms=[J]` is what normalises over the sources for each target while keeping a value per edge, and
    `$h1__att={1, 2}` sits on the *body*, because a head weight would be applied after that aggregation.
    """
    loops = [
        "<1.0> h1__edge(I, I).",
        "h1__edge(I, J) :- *edge(I, J).",
    ]
    normalise = (
        "h1__attention(I, J) :- $h1__att={1, 2} h1__score(I, J). "
        "[transformation=identity, aggregation=softmax(agg_terms=[J])]"
    )
    output = (
        "h1(I) :- h1__attention(I, J), $h1__right={2, 1} h0(J), *h1__edge(J, I). "
        "[combination=product, aggregation=sum]"
    )
    identity = "h1/1 [transformation=identity]"

    def score(left):
        return (
            f"h1__score(I, J) :- ${left}={{2, 1}} h0(I), $h1__right={{2, 1}} h0(J), *h1__edge(J, I). "
            "[transformation=leakyrelu, combination=sum]"
        )

    model = Model()
    model += GATv2Conv(1, 2, "h1", "h0", "_edge")

    assert str(model).split("\n")[:6] == [*loops, score("h1__left"), normalise, output, identity]

    model = Model()
    model += GATv2Conv(1, 2, "h1", "h0", "_edge", share_weights=True)

    assert str(model).split("\n")[:6] == [*loops, score("h1__right"), normalise, output, identity]


def test_sgconv():
    """SGConv adds self-loops and normalises, one factor per hop - PyG's SGConv always does both."""
    setup = [
        "<1.0> h1__edge(I, I).",
        "h1__edge(I, J) :- *edge(I, J).",
        "h1__edge_count(I, J) :- h1__edge(X, J). [aggregation=sum]",
        "h1__edge_count(I, J) :- h1__edge(X, I). [aggregation=sum]",
        "h1__edge_count/2 [transformation=inverse, combination=product]",
    ]
    identity = "h1/1 [transformation=identity]"

    model = Model()
    model += SGConv(1, 2, "h1", "h0", "_edge", k=2)

    assert str(model).split("\n")[:7] == [
        *setup,
        "{2, 1} h1(I0) :- h0(I2), h1__edge(I1, I0), sqrt(h1__edge_count(I1, I0)), h1__edge(I2, I1), "
        "sqrt(h1__edge_count(I2, I1)). [combination=product, aggregation=sum, duplicate_grounding=True]",
        identity,
    ]

    model = Model()
    model += SGConv(1, 2, "h1", "h0", "_edge")

    assert str(model).split("\n")[:7] == [
        *setup,
        "{2, 1} h1(I0) :- h0(I1), h1__edge(I1, I0), sqrt(h1__edge_count(I1, I0)). "
        "[combination=product, aggregation=sum, duplicate_grounding=True]",
        identity,
    ]

    model = Model()
    model += SGConv(1, 2, "h1", "h0", "_edge", normalize=False)

    assert str(model).split("\n")[:2] == [
        "{2, 1} h1(I0) :- h0(I1), *edge(I1, I0). [combination=product, aggregation=sum, duplicate_grounding=True]",
        identity,
    ]


#: APPNP normalises like GCN now, and its edge is a valued atom under a product rather than a hidden one
_APPNP_SETUP = [
    "<1.0> h1__edge(I, I).",
    "h1__edge(I, J) :- *edge(I, J).",
    "h1__edge_count(I, J) :- h1__edge(X, J). [aggregation=sum]",
    "h1__edge_count(I, J) :- h1__edge(X, I). [aggregation=sum]",
    "h1__edge_count/2 [transformation=inverse, combination=product]",
]


def _appnp_teleport(head):
    return f"{head}(I) :- <0.1> h0(I). [combination=product, aggregation=sum]"


def _appnp_propagate(head, source):
    return (
        f"{head}(I) :- <0.9> {source}(J), h1__edge(J, I), sqrt(h1__edge_count(J, I)). "
        "[combination=product, aggregation=sum]"
    )


def test_appnp():
    model = Model()

    model += APPNPConv("h1", "h0", "_edge", 1, 0.1)
    model_str = str(model).split("\n")

    assert model_str[:7] == [*_APPNP_SETUP, _appnp_teleport("h1"), _appnp_propagate("h1", "h0")]

    model = Model()

    model += APPNPConv("h1", "h0", "_edge", 3, 0.1)
    model_str = str(model).split("\n")

    assert model_str[:11] == [
        *_APPNP_SETUP,
        _appnp_teleport("h1__1"),
        _appnp_propagate("h1__1", "h0"),
        _appnp_teleport("h1__2"),
        _appnp_propagate("h1__2", "h1__1"),
        _appnp_teleport("h1"),
        _appnp_propagate("h1", "h1__2"),
    ]

    model = Model()

    model += APPNPConv("h1", "h0", "_edge", 1, 0.1, normalize=False)
    model_str = str(model).split("\n")

    assert model_str[:2] == [
        _appnp_teleport("h1"),
        "h1(I) :- <0.9> h0(J), *edge(J, I). [combination=product, aggregation=sum]",
    ]


def test_res_gated():
    model = Model()

    model += ResGatedGraphConv(1, 2, "h1", "h0", "edge")
    model_str = str(model).split("\n")

    rule = "h1(I) :- h1__gate(I, J), {2, 1} h0(J), edge(J, I). [combination=elproduct, aggregation=sum]"

    assert model_str[0] == "h1__gate(I, J) :- {2, 1} h0(I), {2, 1} h0(J)."
    assert model_str[1] == "h1__gate/2 [transformation=sigmoid]"
    assert model_str[2] == "h1(I) :- {2, 1} h0(I)."
    assert model_str[3] == rule
