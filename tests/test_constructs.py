from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.predicate import PredicateMetadata, Metadata, Predicate
from neuralogic.core import R


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


def test_atom_creation() -> None:
    """Test atom creation related operations and properties"""
    relation = R.my_atom
    assert len(relation.terms) == 0
    assert isinstance(relation, BaseAtom)

    relation = relation("a", "b")
    assert len(relation.terms) == 2
    assert isinstance(relation, BaseAtom)

    relation = relation[1, 2]
    assert len(relation.terms) == 2
    assert isinstance(relation, WeightedAtom)
    assert relation.weight == (1, 2)

    relation = R.my_atom["abc":1, 2]
    assert len(relation.terms) == 0
    assert isinstance(relation, WeightedAtom)
    assert relation.weight == (1, 2)
    assert relation.weight_name == "abc"

    relation = relation.fixed()
    assert relation.is_fixed is True
    assert relation.negated is False

    neg_relation = -relation
    assert neg_relation.negated is True

    neg_relation = ~relation
    assert neg_relation.negated is True
