from neuralogic.core import Model, R, V
from neuralogic.dataset import Dataset, Sample


def test_weighted_rule_query_from_sample_preserves_importance():
    model = Model()
    model += R.out(V.X) <= R.exists(V.X)
    model.build()

    query = R.query_importance[0.25] <= R.out("x")[1.0]
    built_dataset = model.build_dataset(Dataset([Sample(query, [R.exists("x")])]))

    assert len(built_dataset) == 1
    assert built_dataset[0]._java_sample.getImportance() == 0.25


def test_multiple_weighted_rule_queries_can_share_an_example():
    model = Model()
    model += R.out(V.X) <= R.exists(V.X)
    model.build()

    example = [R.exists("x"), R.exists("y")]
    dataset = Dataset(
        [
            Sample(R.query_importance[0.25] <= R.out("x")[1.0], example),
            Sample(R.query_importance[0.75] <= R.out("y")[0.0], example),
        ]
    )
    built_dataset = model.build_dataset(dataset)

    assert len(built_dataset) == 2
    assert [sample._java_sample.getImportance() for sample in built_dataset] == [0.25, 0.75]
