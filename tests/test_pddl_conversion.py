from neuralogic.dataset import PDDLDataset


def test_pddl_conversion() -> None:
    """Test the conversion of PDDL domain and problem strings to a NeuraLogic Dataset."""
    domain_str = """
    (define (domain blocks)
      (:predicates (on ?x ?y) (clear ?x) (holding ?x))
      (:action stack
        :parameters (?x ?y)
        :precondition (and (clear ?y) (holding ?x))
        :effect (and (on ?x ?y) (clear ?x) (not (clear ?y)) (not (holding ?x))))
    )
    """

    problem_str = """
    (define (problem blocks-1)
      (:domain blocks)
      (:objects a b)
      (:init (clear a) (holding b))
      (:goal (on b a))
    )
    """

    pddl_dataset = PDDLDataset(domain_str, problem_str)
    
    logic_dataset = pddl_dataset.to_dataset()

    assert len(logic_dataset) == 1
    sample = logic_dataset[0]

    example_strs = [str(e) for e in sample.example]
    
    assert "clear(a)." in example_strs
    assert "holding(b)." in example_strs
    
    assert "on(X, Y) :- clear(Y), holding(X)." in example_strs
    assert "clear(X) :- clear(Y), holding(X)." in example_strs
    assert "!clear(Y) :- clear(Y), holding(X)." in example_strs
    assert "!holding(X) :- clear(Y), holding(X)." in example_strs

    assert len(sample.query) == 1
    assert str(sample.query[0]) == "on(b, a)."
