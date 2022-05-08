import copy

from neuralogic.core.constructs.relation import BaseRelation, WeightedRelation
from neuralogic.core import R, Activation, Aggregation, Metadata, ActivationAgg, V
from neuralogic.core.constructs.rule import Rule


def test_predicate_creation() -> None:
    """Test different ways of predicate creation"""
    relation = R.lorem_ipsum
    predicate = relation / 1

    assert predicate.arity == 1
    assert predicate.name == "lorem_ipsum"
    assert predicate.special is False
    assert predicate.hidden is False

    predicate = relation / 2
    assert predicate.arity == 2

    assert relation.predicate.arity == 0
    assert relation.predicate.name == "lorem_ipsum"
    assert relation.predicate.special is False
    assert relation.predicate.hidden is False

    relation = R.special.lorem_ipsum
    predicate = relation / 1

    assert predicate.special is True
    assert predicate.hidden is False
    assert relation.predicate.special is True
    assert relation.predicate.hidden is False

    relation = R.hidden.lorem_ipsum
    predicate = relation / 1

    assert predicate.special is False
    assert predicate.hidden is True
    assert relation.predicate.special is False
    assert relation.predicate.hidden is True

    relation = R.special.hidden.lorem_ipsum
    predicate = relation / 1

    assert predicate.special is True
    assert predicate.hidden is True
    assert relation.predicate.special is True
    assert relation.predicate.hidden is True

    relation = R.special._hidden_test
    predicate = relation / 1

    assert predicate.hidden is True
    assert predicate.name == "hidden_test"
    assert predicate.special is True

    relation = R._hidden_test
    predicate = relation / 1

    assert predicate.hidden is True
    assert predicate.special is False
    assert predicate.name == "hidden_test"

    relation = R.get("_hidden_test")
    predicate = relation / 1

    assert predicate.hidden is True
    assert predicate.special is False
    assert predicate.name == "hidden_test"

    predicate_metadata = R.shortest / 2 | [Activation.SIGMOID]
    assert predicate_metadata.metadata is not None
    assert predicate_metadata.metadata.aggregation is None
    assert predicate_metadata.metadata.activation == Activation.SIGMOID

    predicate_metadata = R.shortest / 2 | [Activation.SIGMOID + ActivationAgg.MAX]
    assert predicate_metadata.metadata is not None
    assert predicate_metadata.metadata.aggregation is None
    assert str(predicate_metadata.metadata.activation) == "max-sigmoid"

    predicate_metadata = R.shortest / 2 | [ActivationAgg.MIN]
    assert predicate_metadata.metadata is not None
    assert predicate_metadata.metadata.aggregation is None
    assert str(predicate_metadata.metadata.activation) == "min-identity"

    predicate_metadata = R.shortest / 2 | Metadata(activation=ActivationAgg.MAX + Activation.TANH)
    assert predicate_metadata.metadata is not None
    assert predicate_metadata.metadata.aggregation == None
    assert str(predicate_metadata.metadata.activation) == "max-tanh"


def test_relation_creation() -> None:
    """Test relation creation related operations and properties"""
    relation = R.my_atom
    assert len(relation.terms) == 0
    assert isinstance(relation, BaseRelation)

    relation = relation("a", "b")
    assert len(relation.terms) == 2
    assert isinstance(relation, BaseRelation)

    relation = relation[1, 2]
    assert len(relation.terms) == 2
    assert isinstance(relation, WeightedRelation)
    assert relation.weight == (1, 2)

    relation = R.my_atom["abc":1, 2]
    assert len(relation.terms) == 0
    assert isinstance(relation, WeightedRelation)
    assert relation.weight == (1, 2)
    assert relation.weight_name == "abc"

    relation = relation.fixed()
    assert relation.is_fixed is True
    assert relation.negated is False

    neg_relation = -relation
    assert neg_relation.negated is True

    neg_relation = ~relation
    assert neg_relation.negated is True


def test_rule_metadata():
    rule = (R.a <= R.b) | [Activation.SIGMOID, Aggregation.AVG]

    assert rule.metadata is not None
    assert rule.metadata.aggregation == Aggregation.AVG
    assert rule.metadata.activation == Activation.SIGMOID

    rule = (R.a <= R.b) | Metadata(activation=Activation.IDENTITY, aggregation=Aggregation.MAX)

    assert rule.metadata is not None
    assert rule.metadata.aggregation == Aggregation.MAX
    assert rule.metadata.activation == Activation.IDENTITY

    rule = R.a <= R.b
    assert rule.metadata is None

    rule = (R.a <= R.b) | [ActivationAgg.MAX + Activation.SIGMOID, Aggregation.AVG]

    assert rule.metadata is not None
    assert rule.metadata.aggregation == Aggregation.AVG
    assert str(rule.metadata.activation) == "max-sigmoid"


def test_rules():
    my_rule: Rule = R.a(V.X) <= R.special.alldiff(...)

    assert len(my_rule.body[0].terms) == 1
    assert my_rule.body[0].terms[0] == V.X

    my_rule: Rule = R.a(V.X) <= (R.special.alldiff(...), R.b(V.Y, V.Z))
    assert len(my_rule.body[0].terms) == 3

    terms = sorted(my_rule.body[0].terms)

    assert terms[0] == V.X
    assert terms[1] == V.Y
    assert terms[2] == V.Z

    my_rule = R.a <= R.b

    assert len(my_rule.body) == 1
    assert my_rule.body[0].predicate.name == "b"
