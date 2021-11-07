from neuralogic.core import Template, R, V, T

from neuralogic.core.inference_engine import InferenceEngine


def test_inference_engine_london_reachable() -> None:
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
    expected_yield = [
        {"X": "green_park"},
        {"X": "bond_street"},
    ]

    found_already = [False] * len(expected_yield)

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
    for substitutions in engine.q(R.nearby(V.X, T.oxford_circus)):
        assert "X" in substitutions
        assert len(substitutions) == 1

        for i in range(2):
            if substitutions["X"] == expected_yield[i]["X"]:
                assert not found_already[i]
                found_already[i] = True
    assert all(found_already)

    expected_yield = [
        {"X": "piccadilly_circus", "Z": "piccadilly"},
        {"X": "tottenham_court_road", "Z": "northern"},
    ]

    found_already = [False] * len(expected_yield)

    # Run query for connected(X, leicester_square, Z)
    # Should yield two substitutions:
    # {'X': 'piccadilly_circus', 'Z': 'piccadilly'}, {'X': 'tottenham_court_road', 'Z': 'northern'}
    for substitutions in engine.q(R.connected(V.X, T.leicester_square, V.Z)):
        assert "X" in substitutions
        assert "Z" in substitutions
        assert len(substitutions) == 2

        for i in range(2):
            if substitutions["X"] == expected_yield[i]["X"] and substitutions["Z"] == expected_yield[i]["Z"]:
                assert not found_already[i]
                found_already[i] = True
    assert all(found_already)
