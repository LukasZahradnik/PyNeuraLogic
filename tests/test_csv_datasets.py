import io
from typing import List

import pytest

from neuralogic.dataset import CSVFile, CSVDataset, Mode


def test_csv_file_index_in_header() -> None:
    header = ["a", "b", "c"]

    assert CSVFile._find_index_in_header(header, "a") == 0
    assert CSVFile._find_index_in_header(header, "b") == 1
    assert CSVFile._find_index_in_header(header, "c") == 2

    try:
        CSVFile._find_index_in_header(header, "z")
        assert False
    except ValueError:
        assert True


def tests_csv_file() -> None:
    csv_string_source = """a,b,c
    1,2,3
    6,r,s
    7,8,9
    10,11,12
    13,,
    """

    source = io.StringIO(csv_string_source)
    csv_file = CSVFile("my_rel", source)
    examples = csv_file.to_logic_form()

    expected = [
        "my_rel(a, b, c).",
        "my_rel(1, 2, 3).",
        "my_rel(6, r, s).",
        "my_rel(7, 8, 9).",
        "my_rel(10, 11, 12).",
        "my_rel(13, 0, 0).",
    ]

    assert len(examples) == 6
    assert [str(e) for e in examples] == expected

    source = io.StringIO(csv_string_source)
    csv_file = CSVFile("my_rel", source, default_value=1.0, term_columns=[1, 2], skip_rows=1, n_rows=3)
    examples = csv_file.to_logic_form()

    expected = [
        "1.0 my_rel(2, 3).",
        "1.0 my_rel(r, s).",
        "1.0 my_rel(8, 9).",
    ]

    assert len(examples) == 3
    assert [str(e) for e in examples] == expected

    source = io.StringIO(csv_string_source)
    csv_file = CSVFile("my_rel", source, value_column="a", term_columns=["b", "c"], skip_rows=1, header=True, n_rows=2)
    examples = csv_file.to_logic_form()

    expected = [
        "6.0 my_rel(r, s).",
        "7.0 my_rel(8, 9).",
    ]

    assert len(examples) == 2
    assert [str(e) for e in examples] == expected


@pytest.mark.parametrize(
    "mode,expected",
    [
        (
            Mode.ONE_EXAMPLE,
            [
                [
                    "rel_a(1, 2, 3).",
                    "rel_a(4, 5, 6).",
                    "rel_a(7, 8, 9).",
                    "rel_b(1, 2, 3).",
                    "rel_b(4, 5, 6).",
                    "rel_b(7, 8, 9).",
                    "rel_c(1, 2, 3).",
                    "rel_c(4, 5, 6).",
                    "rel_c(7, 8, 9).",
                ]
            ],
        ),
        (
            Mode.ZIP,
            [
                ["rel_a(1, 2, 3).", "rel_b(1, 2, 3).", "rel_c(1, 2, 3)."],
                ["rel_a(4, 5, 6).", "rel_b(4, 5, 6).", "rel_c(4, 5, 6)."],
                ["rel_a(7, 8, 9).", "rel_b(7, 8, 9).", "rel_c(7, 8, 9)."],
            ],
        ),
        (
            Mode.EXAMPLE_PER_SOURCE,
            [
                [
                    "rel_a(1, 2, 3).",
                    "rel_a(4, 5, 6).",
                    "rel_a(7, 8, 9).",
                ],
                [
                    "rel_b(1, 2, 3).",
                    "rel_b(4, 5, 6).",
                    "rel_b(7, 8, 9).",
                ],
                [
                    "rel_c(1, 2, 3).",
                    "rel_c(4, 5, 6).",
                    "rel_c(7, 8, 9).",
                ],
            ],
        ),
    ],
)
def test_csv_dataset(mode: Mode, expected: List[List[str]]) -> None:
    csv_string_source = """1,2,3
    4,5,6
    7,8,9
    """

    source = io.StringIO(csv_string_source)
    csv_source_a = CSVFile("rel_a", source)

    source = io.StringIO(csv_string_source)
    csv_source_b = CSVFile("rel_b", source)

    source = io.StringIO(csv_string_source)
    csv_source_c = CSVFile("rel_c", source)

    dataset = CSVDataset([csv_source_a, csv_source_b, csv_source_c], mode=mode)
    logic_dataset = dataset.to_dataset()

    assert len(logic_dataset) == len(expected)

    for exp, sample in zip(expected, logic_dataset.samples):
        assert len(exp) == len(sample)
        assert exp == [str(e) for e in sample.example]
