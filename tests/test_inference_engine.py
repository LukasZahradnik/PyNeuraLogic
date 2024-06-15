from neuralogic.core import Template, R, V, C

from neuralogic.inference.inference_engine import InferenceEngine


def test_inference_engine_london_reachable() -> None:
    """
    Test of the inference engine
    based on https://book.simply-logical.space/part_i.html#a_brief_introduction_to_clausal_logic
    """
    template = Template()

    template.add_rules(
        [
            R.connected(C.bond_street, C.oxford_circus, C.central),
            R.connected(C.oxford_circus, C.tottenham_court_road, C.central),
            R.connected(C.bond_street, C.green_park, C.jubilee),
            R.connected(C.green_park, C.charing_cross, C.jubilee),
            R.connected(C.green_park, C.piccadilly_circus, C.piccadilly),
            R.connected(C.piccadilly_circus, C.leicester_square, C.piccadilly),
            R.connected(C.green_park, C.oxford_circus, C.victoria),
            R.connected(C.oxford_circus, C.piccadilly_circus, C.bakerloo),
            R.connected(C.piccadilly_circus, C.charing_cross, C.bakerloo),
            R.connected(C.tottenham_court_road, C.leicester_square, C.northern),
            R.connected(C.leicester_square, C.charing_cross, C.northern),
            R.reachable(V.X, V.Y) <= R.connected(V.X, V.Y, V.L),
            R.reachable(V.X, V.Y) <= (R.connected(V.X, V.Z, V.L), R.reachable(V.Z, V.Y)),
        ]
    )

    engine = InferenceEngine(template)

    # ask if tottenham_court_road can be reached from green_park
    assert engine.query(R.reachable(C.green_park, C.tottenham_court_road))

    # green_park cannot be reached from charing_cross
    # random_place does not exist in the dataset (cannot be reached from anywhere)
    assert not engine.query(R.reachable(C.charing_cross, C.green_park))
    assert not engine.q(R.reachable(C.charing_cross, C.random_place))


def test_inference_engine_london() -> None:
    """
    Test of the inference engine
    based on https://book.simply-logical.space/part_i.html#a_brief_introduction_to_clausal_logic
    """
    template = Template()

    knowledge = [
        R.connected(C.bond_street, C.oxford_circus, C.central),
        R.connected(C.oxford_circus, C.tottenham_court_road, C.central),
        R.connected(C.bond_street, C.green_park, C.jubilee),
        R.connected(C.green_park, C.charing_cross, C.jubilee),
        R.connected(C.green_park, C.piccadilly_circus, C.piccadilly),
        R.connected(C.piccadilly_circus, C.leicester_square, C.piccadilly),
        R.connected(C.green_park, C.oxford_circus, C.victoria),
        R.connected(C.oxford_circus, C.piccadilly_circus, C.bakerloo),
        R.connected(C.piccadilly_circus, C.charing_cross, C.bakerloo),
        R.connected(C.tottenham_court_road, C.leicester_square, C.northern),
        R.connected(C.leicester_square, C.charing_cross, C.northern),
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
    substitutions = list(engine.q(R.nearby(V.X, C.oxford_circus)))

    assert substitutions[0]["X"] == "bond_street"
    assert substitutions[1]["X"] == "green_park"
    assert len(substitutions) == 2 and len(substitutions[0]) == 1 and len(substitutions[1]) == 1

    # Run query for nearby(X, tottenham_court_road)
    substitutions = sorted(engine.q(R.nearby(V.X, C.charing_cross)), key=lambda x: x["X"])

    assert substitutions[0]["X"] == "bond_street"
    assert substitutions[1]["X"] == "green_park"
    assert substitutions[2]["X"] == "leicester_square"
    assert substitutions[3]["X"] == "oxford_circus"
    assert substitutions[4]["X"] == "piccadilly_circus"
    assert substitutions[5]["X"] == "tottenham_court_road"
    assert len(substitutions) == 6 and len(substitutions[0]) == 1 and len(substitutions[1]) == 1

    # Run query for connected(X, leicester_square, Z)
    # Should yield two substitutions:
    # {'X': 'piccadilly_circus', 'Z': 'piccadilly'}, {'X': 'tottenham_court_road', 'Z': 'northern'}
    substitutions = list(engine.q(R.connected(V.X, C.leicester_square, V.Z)))

    assert substitutions[0]["X"] == "piccadilly_circus"
    assert substitutions[0]["Z"] == "piccadilly"

    assert substitutions[1]["X"] == "tottenham_court_road"
    assert substitutions[1]["Z"] == "northern"
    assert len(substitutions) == 2 and len(substitutions[0]) == 2 and len(substitutions[1]) == 2


def test_listing_all_queries() -> None:
    template = Template()

    template += R.h(V.X) <= R.edge(V.Y, V.X)
    template += R.h1(V.X) <= (R.h(V.Y), R.edge(V.Y, V.X))
    template += R.q <= R.h1(V.X)

    template += R.edge(1, 2)
    template += R.edge(2, 3)
    template += R.edge(3, 1)

    inference_engine = InferenceEngine(template)

    queries = list(inference_engine.get_queries())

    expected_queries = sorted(["h(2).", "h(3).", "h(1).", "h1(1).", "h1(2).", "h1(3).", "q."])
    str_queries = sorted([str(query) for query in queries])

    for a, b in zip(expected_queries, str_queries):
        assert a == b

    queries = list(inference_engine.get_queries([R.edge(1, 4)]))

    expected_queries = sorted(["h(2).", "h(3).", "h(1).", "h(4).", "h1(1).", "h1(2).", "h1(3).", "h1(4).", "q."])
    str_queries = sorted([str(query) for query in queries])

    for a, b in zip(expected_queries, str_queries):
        assert a == b
