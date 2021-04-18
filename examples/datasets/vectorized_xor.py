from neuralogic.model import Atom, Model
from neuralogic.settings import Settings, Optimizer

settings = Settings(optimizer=Optimizer.SGD, epochs=300)


with Model(settings).context() as model:
    model.add_rule(Atom.xor[1, 8] <= Atom.xy[8, 2])  # Add template rule

    model.add_examples(
        [  # Add 4 examples
            Atom.xor[0] <= Atom.xy[[0, 0]],
            Atom.xor[1] <= Atom.xy[[0, 1]],
            Atom.xor[1] <= Atom.xy[[1, 0]],
            Atom.xor[0] <= Atom.xy[[1, 1]],
        ]
    )
