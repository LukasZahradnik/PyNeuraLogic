from neuralogic.core import Template, R, V, C, Metadata, Aggregation, Transformation

from neuralogic.inference.evaluation_inference_engine import EvaluationInferenceEngine


def test_evaluation_inference_engine_london_reachable() -> None:
    """
    Test of the evaluation inference engine
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

    engine = EvaluationInferenceEngine(template)

    # ask if tottenham_court_road can be reached from green_park
    assert engine.query(R.reachable(C.green_park, C.tottenham_court_road))

    # green_park cannot be reached from charing_cross
    # random_place does not exist in the dataset (cannot be reached from anywhere)
    assert not engine.query(R.reachable(C.charing_cross, C.green_park))
    assert not engine.q(R.reachable(C.charing_cross, C.random_place))


def test_evaluation_inference_engine_london() -> None:
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

    engine = EvaluationInferenceEngine(template)
    engine.set_knowledge(knowledge)

    # Run query for nearby(X, oxford_circus)
    # Should yield two substitutions for x (green_park and bond_street)
    substitutions = list(engine.q(R.nearby(V.X, C.oxford_circus)))
    assert len(substitutions) == 2 and len(substitutions[0][1]) == 1 and len(substitutions[1][1]) == 1

    variable_subs = {substitutions[i][1]["X"] for i in range(2)}
    assert len(variable_subs) == 2
    assert "green_park" in variable_subs
    assert "bond_street" in variable_subs

    # Run query for nearby(X, tottenham_court_road)
    substitutions = list(engine.q(R.nearby(V.X, C.charing_cross)))

    assert len(substitutions) == 6 and len(substitutions[0][1]) == 1 and len(substitutions[1][1]) == 1

    variable_subs = {substitutions[i][1]["X"] for i in range(6)}
    assert len(variable_subs) == 6

    assert "piccadilly_circus" in variable_subs
    assert "leicester_square" in variable_subs
    assert "green_park" in variable_subs
    assert "oxford_circus" in variable_subs
    assert "tottenham_court_road" in variable_subs
    assert "bond_street" in variable_subs

    # Run query for connected(X, leicester_square, Z)
    # Should yield two substitutions:
    # {'X': 'piccadilly_circus', 'Z': 'piccadilly'}, {'X': 'tottenham_court_road', 'Z': 'northern'}
    # substitutions = list(engine.q(R.connected(T.piccadilly_circus, T.leicester_square, T.piccadilly)))
    # assert len(substitutions) == 2 and len(substitutions[0][1]) == 2 and len(substitutions[1][1]) == 2
    #
    # if substitutions[0][1]["X"] == "piccadilly_circus":
    #     assert substitutions[0][1]["Z"] == "piccadilly"
    #
    #     assert substitutions[1][1]["X"] == "tottenham_court_road"
    #     assert substitutions[1][1]["Z"] == "northern"
    # else:
    #     assert substitutions[1][1]["X"] == "piccadilly_circus"
    #     assert substitutions[1][1]["Z"] == "piccadilly"
    #
    #     assert substitutions[0][1]["X"] == "tottenham_court_road"
    #     assert substitutions[0][1]["Z"] == "northern"


def test_evaluation_inference_engine_london_shortest_path() -> None:
    """
    Test of the inference engine for finding shortest paths
    based on https://book.simply-logical.space/part_i.html#a_brief_introduction_to_clausal_logic
    """
    template = Template()

    knowledge = [
        R.connected(C.bond_street, C.oxford_circus, C.central)[-7],
        R.connected(C.oxford_circus, C.tottenham_court_road, C.central)[-9],
        R.connected(C.bond_street, C.green_park, C.jubilee)[-14],
        R.connected(C.green_park, C.charing_cross, C.jubilee)[-21],
        R.connected(C.green_park, C.piccadilly_circus, C.piccadilly)[-8],
        R.connected(C.piccadilly_circus, C.leicester_square, C.piccadilly)[-6],
        R.connected(C.green_park, C.oxford_circus, C.victoria)[-15],
        R.connected(C.oxford_circus, C.piccadilly_circus, C.bakerloo)[-12],
        R.connected(C.piccadilly_circus, C.charing_cross, C.bakerloo)[-11],
        R.connected(C.tottenham_court_road, C.leicester_square, C.northern)[-8],
        R.connected(C.leicester_square, C.charing_cross, C.northern)[-7],
    ]

    metadata = Metadata(aggregation=Aggregation.MAX, transformation=Transformation.IDENTITY)

    template += [
        (R.shortest(V.X, V.Y, C.first) <= R.connected(V.X, V.Y, V.L)) | metadata,
        (R.shortest(V.X, V.Y, C.second) <= (R.connected(V.X, V.Z, V.L), R.shortest(V.Z, V.Y, V.D))) | metadata,
        (R.shortest_path(V.X, V.Y) <= R.shortest(V.X, V.Y, V.D)) | metadata,
        R.shortest / 3 | Metadata(transformation=Transformation.IDENTITY),
        R.connected / 3 | Metadata(transformation=Transformation.IDENTITY),
        R.shortest_path / 2 | Metadata(transformation=Transformation.IDENTITY),
    ]

    engine = EvaluationInferenceEngine(template)
    engine.set_knowledge(knowledge)

    # The shortest path from Bond Street to Oxford Street is 7
    substitutions = list(engine.q(R.shortest_path(C.bond_street, C.oxford_circus)))
    assert len(substitutions) == 1
    assert len(substitutions[0][1]) == 0
    assert substitutions[0][0] == -7

    # The shortest path from Bond Street to Oxford Street is 24 (Bond Street -> Oxford Circus -> Tottenham -> Leicester)
    substitutions = list(engine.q(R.shortest_path(C.bond_street, C.leicester_square)))
    assert len(substitutions) == 1
    assert len(substitutions[0][1]) == 0
    assert substitutions[0][0] == -24

    # Shortest paths from Green Park to every reachable station
    substitutions = list(engine.q(R.shortest_path(C.green_park, V.X)))
    assert len(substitutions) == 5

    expected_results = {
        "oxford_circus": -15,
        "tottenham_court_road": -24,
        "piccadilly_circus": -8,
        "leicester_square": -14,
        "charing_cross": -19,
    }

    all_substitutions = set()

    for length, substitution in substitutions:
        assert len(substitution) == 1
        assert length == expected_results[substitution["X"]]

        all_substitutions.add(substitution["X"])

    assert len(all_substitutions) == 5
