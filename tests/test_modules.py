from neuralogic.core import Template, Transformation, Aggregation
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
    template = Template()

    template += RGCNConv(1, 2, "h1", "h0", "_edge", ["a", "b", "c"])
    template_str = str(template).split("\n")

    assert template_str[0] == "h1(I) :- {2, 1} h0(I). [aggregation=avg]"
    assert template_str[1] == "h1(I) :- {2, 1} h0(J), *edge(J, a, I). [aggregation=avg]"
    assert template_str[2] == "h1(I) :- {2, 1} h0(J), *edge(J, b, I). [aggregation=avg]"
    assert template_str[3] == "h1(I) :- {2, 1} h0(J), *edge(J, c, I). [aggregation=avg]"
    assert template_str[4] == "h1/1 [transformation=identity]"


def test_rgcnconv_relations_edge_replace():
    template = Template()

    template += RGCNConv(1, 2, "h1", "h0", None, ["a", "b", "c"], Transformation.SIGMOID)
    template_str = str(template).split("\n")

    assert template_str[0] == "h1(I) :- {2, 1} h0(I). [aggregation=avg]"
    assert template_str[1] == "h1(I) :- {2, 1} h0(J), a(J, I). [aggregation=avg]"
    assert template_str[2] == "h1(I) :- {2, 1} h0(J), b(J, I). [aggregation=avg]"
    assert template_str[3] == "h1(I) :- {2, 1} h0(J), c(J, I). [aggregation=avg]"
    assert template_str[4] == "h1/1 [transformation=sigmoid]"


def test_gcnconv():
    template = Template()

    template += GCNConv(1, 2, "h1", "h0", "edge")
    template_str = str(template).split("\n")

    assert template_str[0] == "<1.0> h1__edge(I, I)."
    assert template_str[1] == "h1__edge(I, J) :- edge(I, J)."
    assert template_str[2] == "h1__edge_count(I, J) :- h1__edge(J, X). [aggregation=count]"
    assert template_str[3] == "h1__edge_count(I, J) :- h1__edge(I, X). [aggregation=count]"
    assert template_str[4] == "h1__edge_count/2 [transformation=inverse, combination=product]"
    assert (
        template_str[5]
        == "{2, 1} h1(I) :- h0(J), h1__edge(J, I), sqrt(h1__edge_count(J, I)). [combination=product, aggregation=sum]"
    )


def test_sageconv():
    template = Template()

    template += SAGEConv(1, 2, "h1", "h0", "_edge")
    template_str = str(template).split("\n")

    assert template_str[0] == "{2, 1} h1(I) :- h0(J), *edge(J, I). [aggregation=avg]"
    assert template_str[1] == "{2, 1} h1(I) :- h0(I). [aggregation=avg]"


def test_tagconv():
    template = Template()

    template += TAGConv(1, 2, "h1", "h0", "_edge")
    template_str = str(template).split("\n")

    zero_hop = "{2, 1} h1(I0) :- h0(I0). [aggregation=sum]"
    sec_hop = "{2, 1} h1(I0) :- h0(I1), *edge(I1, I0). [aggregation=sum]"
    hop = "{2, 1} h1(I0) :- h0(I2), *edge(I1, I0), *edge(I2, I1). [aggregation=sum]"

    assert template_str[0] == zero_hop
    assert template_str[1] == sec_hop
    assert template_str[2] == hop

    template = Template()

    template += TAGConv(1, 2, "h1", "h0", "_edge", 1)
    template_str = str(template).split("\n")

    assert template_str[0] == zero_hop
    assert template_str[1] == sec_hop
    assert template_str[2] == "h1/1 [transformation=identity]"


def test_gatv2conv():
    template = Template()

    template += GATv2Conv(1, 2, "h1", "h0", "_edge")
    template_str = str(template).split("\n")

    attention = (
        "{2, 2} h1__attention(I, J) :- $h1__left={2, 1} h0(I), $h1__right={2, 1} h0(J). [transformation=leakyrelu]"
    )
    assert template_str[0] == attention
    assert template_str[1] == "h1__attention/2 [transformation=softmax]"

    h1_rule = (
        "h1(I) :- h1__attention(I, J), $h1__right={2, 1} h0(J), *edge(J, I). [combination=product, aggregation=sum]"
    )
    assert template_str[2] == h1_rule

    template = Template()

    template += GATv2Conv(1, 2, "h1", "h0", "_edge", share_weights=True)
    template_str = str(template).split("\n")

    attention = (
        "{2, 2} h1__attention(I, J) :- $h1__right={2, 1} h0(I), $h1__right={2, 1} h0(J). [transformation=leakyrelu]"
    )
    assert template_str[0] == attention
    assert template_str[1] == "h1__attention/2 [transformation=softmax]"

    h1_rule = (
        "h1(I) :- h1__attention(I, J), $h1__right={2, 1} h0(J), *edge(J, I). [combination=product, aggregation=sum]"
    )
    assert template_str[2] == h1_rule


def test_sgconv():
    template = Template()

    template += SGConv(1, 2, "h1", "h0", "_edge", k=2)
    template_str = str(template).split("\n")
    rule = "{2, 1} h1(I0) :- h0(I2), *edge(I1, I0), *edge(I2, I1). [aggregation=sum, duplicit_grounding=True]"

    assert template_str[0] == rule

    template = Template()

    template += SGConv(1, 2, "h1", "h0", "_edge")
    template_str = str(template).split("\n")
    rule = "{2, 1} h1(I0) :- h0(I1), *edge(I1, I0). [aggregation=sum, duplicit_grounding=True]"

    assert template_str[0] == rule


def test_appnp():
    template = Template()

    template += APPNPConv("h1", "h0", "_edge", 1, 0.1)
    template_str = str(template).split("\n")

    assert template_str[0] == "h1(I) :- <0.1> h0(I). [aggregation=sum]"
    assert template_str[1] == "h1(I) :- <0.9> h0(J), *edge(J, I). [aggregation=sum]"

    template = Template()

    template += APPNPConv("h1", "h0", "_edge", 3, 0.1)
    template_str = str(template).split("\n")

    assert template_str[0] == "h1__1(I) :- <0.1> h0(I). [aggregation=sum]"
    assert template_str[1] == "h1__1(I) :- <0.9> h0(J), *edge(J, I). [aggregation=sum]"

    assert template_str[2] == "h1__2(I) :- <0.1> h0(I). [aggregation=sum]"
    assert template_str[3] == "h1__2(I) :- <0.9> h1__1(J), *edge(J, I). [aggregation=sum]"

    assert template_str[4] == "h1(I) :- <0.1> h0(I). [aggregation=sum]"
    assert template_str[5] == "h1(I) :- <0.9> h1__2(J), *edge(J, I). [aggregation=sum]"


def test_res_gated():
    template = Template()

    template += ResGatedGraphConv(1, 2, "h1", "h0", "edge")
    template_str = str(template).split("\n")

    rule = "h1(I) :- h1__gate(I, J), {2, 1} h0(J), edge(J, I). [combination=elproduct, aggregation=sum]"

    assert template_str[0] == "h1__gate(I, J) :- {2, 1} h0(I), {2, 1} h0(J)."
    assert template_str[1] == "h1__gate/2 [transformation=sigmoid]"
    assert template_str[2] == "h1(I) :- {2, 1} h0(I)."
    assert template_str[3] == rule
