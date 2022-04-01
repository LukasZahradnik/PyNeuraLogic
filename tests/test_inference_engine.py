from neuralogic.core import Template, R, V, T

from neuralogic.inference.inference_engine import InferenceEngine


def test_inference_engine_london_reachable() -> None:
    """
    Test of the inference engine
    based on https://book.simply-logical.space/part_i.html#a_brief_introduction_to_clausal_logic
    """
    template = Template()

    template.add_rules(
        [
            R.connected(T.bond_street, T.oxford_circus, T.central),
            R.connected(T.oxford_circus, T.tottenham_court_road, T.central),
            R.connected(T.bond_street, T.green_park, T.jubilee),
            R.connected(T.green_park, T.charing_cross, T.jubilee),
            R.connected(T.green_park, T.piccadilly_circus, T.piccadilly),
            R.connected(T.piccadilly_circus, T.leicester_square, T.piccadilly),
            R.connected(T.green_park, T.oxford_circus, T.victoria),
            R.connected(T.oxford_circus, T.piccadilly_circus, T.bakerloo),
            R.connected(T.piccadilly_circus, T.charing_cross, T.bakerloo),
            R.connected(T.tottenham_court_road, T.leicester_square, T.northern),
            R.connected(T.leicester_square, T.charing_cross, T.northern),
            R.reachable(V.X, V.Y) <= R.connected(V.X, V.Y, V.L),
            R.reachable(V.X, V.Y) <= (R.connected(V.X, V.Z, V.L), R.reachable(V.Z, V.Y)),
        ]
    )

    engine = InferenceEngine(template)

    # ask if tottenham_court_road can be reached from green_park
    assert engine.query(R.reachable(T.green_park, T.tottenham_court_road))

    # green_park cannot be reached from charing_cross
    # random_place does not exist in the dataset (cannot be reached from anywhere)
    assert not engine.query(R.reachable(T.charing_cross, T.green_park))
    assert not engine.q(R.reachable(T.charing_cross, T.random_place))


def test_inference_engine_london() -> None:
    """
    Test of the inference engine
    based on https://book.simply-logical.space/part_i.html#a_brief_introduction_to_clausal_logic
    """
    template = Template()

    knowledge = [
        R.connected(T.bond_street, T.oxford_circus, T.central),
        R.connected(T.oxford_circus, T.tottenham_court_road, T.central),
        R.connected(T.bond_street, T.green_park, T.jubilee),
        R.connected(T.green_park, T.charing_cross, T.jubilee),
        R.connected(T.green_park, T.piccadilly_circus, T.piccadilly),
        R.connected(T.piccadilly_circus, T.leicester_square, T.piccadilly),
        R.connected(T.green_park, T.oxford_circus, T.victoria),
        R.connected(T.oxford_circus, T.piccadilly_circus, T.bakerloo),
        R.connected(T.piccadilly_circus, T.charing_cross, T.bakerloo),
        R.connected(T.tottenham_court_road, T.leicester_square, T.northern),
        R.connected(T.leicester_square, T.charing_cross, T.northern),
    ]

    template.add_rules(
        [
            R.nearby(V.X, V.Y) <= R.connected(V.X, V.Y, V.L),
            R.nearby(V.X, V.Y) <= (R.connected(V.X, V.Z, V.L), R.connected(V.Z, V.Y, V.L)),
        ]
    )

    engine = InferenceEngine(template)
    engine.set_knowledge(knowledge)

    # Run query for nearby(X, oxford_circus)
    # Should yield two substitutions for x (green_park and bond_street)
    substitutions = list(engine.q(R.nearby(V.X, T.oxford_circus)))

    assert substitutions[0]["X"] == "green_park"
    assert substitutions[1]["X"] == "bond_street"
    assert len(substitutions) == 2 and len(substitutions[0]) == 1 and len(substitutions[1]) == 1

    # Run query for nearby(X, tottenham_court_road)
    substitutions = list(engine.q(R.nearby(V.X, T.charing_cross)))

    assert substitutions[0]["X"] == "piccadilly_circus"
    assert substitutions[1]["X"] == "leicester_square"
    assert substitutions[2]["X"] == "tottenham_court_road"
    assert substitutions[3]["X"] == "green_park"
    assert substitutions[4]["X"] == "bond_street"
    assert substitutions[5]["X"] == "oxford_circus"
    assert len(substitutions) == 6 and len(substitutions[0]) == 1 and len(substitutions[1]) == 1

    # Run query for connected(X, leicester_square, Z)
    # Should yield two substitutions:
    # {'X': 'piccadilly_circus', 'Z': 'piccadilly'}, {'X': 'tottenham_court_road', 'Z': 'northern'}
    substitutions = list(engine.q(R.connected(V.X, T.leicester_square, V.Z)))

    assert substitutions[0]["X"] == "piccadilly_circus"
    assert substitutions[0]["Z"] == "piccadilly"

    assert substitutions[1]["X"] == "tottenham_court_road"
    assert substitutions[1]["Z"] == "northern"
    assert len(substitutions) == 2 and len(substitutions[0]) == 2 and len(substitutions[1]) == 2
