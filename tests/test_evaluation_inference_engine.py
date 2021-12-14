from neuralogic.core import Template, R, V, T, Metadata, Aggregation, Activation

from neuralogic.core.evaluation_inference_engine import EvaluationInferenceEngine


def test_eval_inference_engine_london_reachable() -> None:
    """
    Test of the evaluation inference engine
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

    engine = EvaluationInferenceEngine(template)

    # ask if tottenham_court_road can be reached from green_park
    assert engine.query(R.reachable(T.green_park, T.tottenham_court_road))

    # green_park cannot be reached from charing_cross
    # random_place does not exist in the dataset (cannot be reached from anywhere)
    assert not engine.query(R.reachable(T.charing_cross, T.green_park))
    assert not engine.q(R.reachable(T.charing_cross, T.random_place))


def test_evaluation_inference_engine_london() -> None:
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

    engine = EvaluationInferenceEngine(template)
    engine.set_knowledge(knowledge)

    # Run query for nearby(X, oxford_circus)
    # Should yield two substitutions for x (green_park and bond_street)
    substitutions = list(engine.q(R.nearby(V.X, T.oxford_circus)))
    assert len(substitutions) == 2 and len(substitutions[0][1]) == 1 and len(substitutions[1][1]) == 1

    variable_subs = {substitutions[i][1]["X"] for i in range(2)}
    assert len(variable_subs) == 2
    assert "green_park" in variable_subs
    assert "bond_street" in variable_subs

    # Run query for nearby(X, tottenham_court_road)
    substitutions = list(engine.q(R.nearby(V.X, T.charing_cross)))

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
        R.connected(T.bond_street, T.oxford_circus, T.central)[-7],
        R.connected(T.oxford_circus, T.tottenham_court_road, T.central)[-9],
        R.connected(T.bond_street, T.green_park, T.jubilee)[-14],
        R.connected(T.green_park, T.charing_cross, T.jubilee)[-21],
        R.connected(T.green_park, T.piccadilly_circus, T.piccadilly)[-8],
        R.connected(T.piccadilly_circus, T.leicester_square, T.piccadilly)[-6],
        R.connected(T.green_park, T.oxford_circus, T.victoria)[-15],
        R.connected(T.oxford_circus, T.piccadilly_circus, T.bakerloo)[-12],
        R.connected(T.piccadilly_circus, T.charing_cross, T.bakerloo)[-11],
        R.connected(T.tottenham_court_road, T.leicester_square, T.northern)[-8],
        R.connected(T.leicester_square, T.charing_cross, T.northern)[-7],
    ]

    metadata = Metadata(aggregation=Aggregation.MAX, activation=Activation.IDENTITY)

    template += [
        (R.shortest(V.X, V.Y, T.first) <= R.connected(V.X, V.Y, V.L)) | metadata,
        (R.shortest(V.X, V.Y, T.second) <= (R.connected(V.X, V.Z, V.L), R.shortest(V.Z, V.Y, V.D))) | metadata,
        (R.shortest_path(V.X, V.Y) <= R.shortest(V.X, V.Y, V.D)) | metadata,
        R.shortest / 3 | Metadata(activation=Activation.IDENTITY),
        R.connected / 3 | Metadata(activation=Activation.IDENTITY),
        R.shortest_path / 2 | Metadata(activation=Activation.IDENTITY),
    ]

    engine = EvaluationInferenceEngine(template)
    engine.set_knowledge(knowledge)

    # The shortest path from Bond Street to Oxford Street is 7
    substitutions = list(engine.q(R.shortest_path(T.bond_street, T.oxford_circus)))
    assert len(substitutions) == 1
    assert len(substitutions[0][1]) == 0
    assert substitutions[0][0] == -7

    # The shortest path from Bond Street to Oxford Street is 24 (Bond Street -> Oxford Circus -> Tottenham -> Leicester)
    substitutions = list(engine.q(R.shortest_path(T.bond_street, T.leicester_square)))
    assert len(substitutions) == 1
    assert len(substitutions[0][1]) == 0
    assert substitutions[0][0] == -24

    # Shortest paths from Green Park to every reachable station
    substitutions = list(engine.q(R.shortest_path(T.green_park, V.X)))
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
