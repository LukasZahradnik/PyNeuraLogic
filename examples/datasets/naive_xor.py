from neuralogic.core import Relation, Template, Transformation
from neuralogic.dataset import Dataset, Sample


dataset = Dataset()
template = Template()

# fmt: off

# hidden<1-8> :- {1} a, {1} b.
template.add_rules([(Relation.get(f"hidden{i}") <= (Relation.a[1, ], Relation.b[1, ])) | [Transformation.TANH] for i in range(1, 9)])
template.add_rules([Relation.get(f"hidden{i}") / 0 | [Transformation.TANH] for i in range(1, 9)])

# {1} xor :- hidden<1-8>.
template.add_rules([(Relation.xor[1, ] <= Relation.get(f"hidden{i}")) | [Transformation.TANH] for i in range(1, 9)])
template.add_rules([Relation.xor / 0 | [Transformation.TANH] for i in range(1, 9)])

dataset.add_samples(
    [  # Add 4 examples
        Sample(Relation.xor[0], [Relation.a[0], Relation.b[0]]),
        Sample(Relation.xor[1], [Relation.a[1], Relation.b[0]]),
        Sample(Relation.xor[1], [Relation.a[0], Relation.b[1]]),
        Sample(Relation.xor[0], [Relation.a[1], Relation.b[1]]),
    ]
)
