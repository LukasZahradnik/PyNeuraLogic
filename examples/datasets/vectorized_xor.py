from neuralogic.core import Relation, Template, Transformation
from neuralogic.dataset import Dataset, Sample


dataset = Dataset()


template = Template()

template.add_rule((Relation.xor[1, 8] <= Relation.xy[8, 2]) | [Transformation.TANH])  # Add template rule
template.add_rule(Relation.xor / 0 | [Transformation.TANH])

dataset.add_samples(
    [  # Add 4 examples
        Sample(Relation.xor[0], Relation.xy[[0, 0]]),
        Sample(Relation.xor[1], Relation.xy[[0, 1]]),
        Sample(Relation.xor[1], Relation.xy[[1, 0]]),
        Sample(Relation.xor[0], Relation.xy[[1, 1]]),
    ]
)
