from neuralogic.core import Atom, Template
from neuralogic.core.settings import Settings, Optimizer
from neuralogic.utils.data import Dataset


settings = Settings(optimizer=Optimizer.SGD)
dataset = Dataset()


with Template(settings).context() as template:
    template.add_rule(Atom.xor[1, 8] <= Atom.xy[8, 2])  # Add template rule

    dataset.add_examples(
        [  # Add 4 examples
            Atom.xor[0] <= Atom.xy[[0, 0]],
            Atom.xor[1] <= Atom.xy[[0, 1]],
            Atom.xor[1] <= Atom.xy[[1, 0]],
            Atom.xor[0] <= Atom.xy[[1, 1]],
        ]
    )
