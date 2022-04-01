from neuralogic.core import Relation, Template
from neuralogic.dataset import Dataset


dataset = Dataset()


template = Template()

template.add_rule(Relation.xor[1, 8] <= Relation.xy[8, 2])  # Add template rule

dataset.add_examples(
    [  # Add 4 examples
        Relation.xor[0] <= Relation.xy[[0, 0]],
        Relation.xor[1] <= Relation.xy[[0, 1]],
        Relation.xor[1] <= Relation.xy[[1, 0]],
        Relation.xor[0] <= Relation.xy[[1, 1]],
    ]
)
