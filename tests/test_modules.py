from neuralogic.core import Template, Activation, Aggregation
from neuralogic.utils.module import RGCNConv, SAGEConv, GCNConv, GINConv


def test_rgcnconv():
    template = Template()

    template += RGCNConv(1, 2, "h1", "h0", "_edge", ["a", "b", "c"])
    template_str = str(template).split("\n")

    assert template_str[0] == "h1(X) :- {2, 1} h0(X). [activation=identity, aggregation=avg]"
    assert template_str[1] == "h1(X) :- {2, 1} h0(Y), *edge(Y, a, X). [activation=identity, aggregation=avg]"
    assert template_str[2] == "h1(X) :- {2, 1} h0(Y), *edge(Y, b, X). [activation=identity, aggregation=avg]"
    assert template_str[3] == "h1(X) :- {2, 1} h0(Y), *edge(Y, c, X). [activation=identity, aggregation=avg]"
    assert template_str[4] == "h1/1 [activation=identity]"


def test_rgcnconv_relations_edge_replace():
    template = Template()

    template += RGCNConv(1, 2, "h1", "h0", None, ["a", "b", "c"], Activation.SIGMOID)
    template_str = str(template).split("\n")

    assert template_str[0] == "h1(X) :- {2, 1} h0(X). [activation=identity, aggregation=avg]"
    assert template_str[1] == "h1(X) :- {2, 1} h0(Y), a(Y, X). [activation=identity, aggregation=avg]"
    assert template_str[2] == "h1(X) :- {2, 1} h0(Y), b(Y, X). [activation=identity, aggregation=avg]"
    assert template_str[3] == "h1(X) :- {2, 1} h0(Y), c(Y, X). [activation=identity, aggregation=avg]"
    assert template_str[4] == "h1/1 [activation=sigmoid]"


def test_gcnconv():
    template = Template()

    template += GCNConv(1, 2, "h1", "h0", "_edge")
    template_str = str(template).split("\n")

    assert template_str[0] == "{2, 1} h1(X) :- h0(Y), *edge(Y, X). [activation=identity, aggregation=sum]"
    assert template_str[1] == "h1/1 [activation=identity]"


def test_sageconv():
    template = Template()

    template += SAGEConv(1, 2, "h1", "h0", "_edge")
    template_str = str(template).split("\n")

    assert template_str[0] == "{2, 1} h1(X) :- h0(Y), *edge(Y, X). [activation=identity, aggregation=sum]"
    assert template_str[1] == "{2, 1} h1(X) :- h0(X). [activation=identity, aggregation=sum]"
    assert template_str[2] == "h1/1 [activation=identity]"
