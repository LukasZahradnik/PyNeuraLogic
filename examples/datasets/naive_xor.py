from neuralogic.core import Relation, Template
from neuralogic.core.settings import Settings, Optimizer
from neuralogic.utils.data import Dataset

settings = Settings(optimizer=Optimizer.SGD, epochs=300)
dataset = Dataset()
template = Template(settings)

# fmt: off

# hidden<1-8> :- {1} a, {1} b.
template.add_rules([Relation.get(f"hidden{i}") <= (Relation.a[1,], Relation.b[1,]) for i in range(1, 9)])

# {1} xor :- hidden<1-8>.
template.add_rules([Relation.xor[1,] <= Relation.get(f"hidden{i}") for i in range(1, 9)])

dataset.add_examples(
    [  # Add 4 examples
        Relation.xor[0] <= (Relation.a[0], Relation.b[0]),
        Relation.xor[1] <= (Relation.a[1], Relation.b[0]),
        Relation.xor[1] <= (Relation.a[0], Relation.b[1]),
        Relation.xor[0] <= (Relation.a[1], Relation.b[1]),
    ]
)
