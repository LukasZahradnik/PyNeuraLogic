from neuralogic.model import Atom, Model

with Model().context() as model:
    model.add_rule(Atom.xor[1, 8] <= Atom.xy[8, 2])  # Add template rule

    model.add_examples(
        [  # Add 4 examples
            Atom.xor[0] <= Atom.xy[[0, 0]],
            Atom.xor[1] <= Atom.xy[[0, 1]],
            Atom.xor[1] <= Atom.xy[[1, 0]],
            Atom.xor[0] <= Atom.xy[[1, 1]],
        ]
    )

    dataset = model.build()  # Build model into dataset (weights and samples)
