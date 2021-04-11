from neuralogic.model import Atom, Model, Var, Term, Metadata, Activation
from examples.datasets.data.train_example_data import train_example_data
from typing import List


with Model().context() as model:
    # One example per train, doesn't know order of vagons

    # fmt: off

    shapes = [Term.ellipse, Term.rectangle, Term.bucket, Term.hexagon, Term.u_shaped]
    roofs = [Term.jagged, Term.arc, Term.none, Term.flat, Term.peaked]
    loadshapes = [Term.hexagon, Term.triangle, Term.diamond, Term.rectangle, Term.circle]
    vagon_atoms = [Atom.shape, Atom.length, Atom.sides, Atom.wheels, Atom.loadnum, Atom.loadshape, Atom.roof]

    X = Var.X
    Y = Var.Y

    model.add_rules(
        [
            *[Atom.shape(Y) <= Atom.shape(X, Y, s)[1,] for s in shapes],
            *[Atom.length(Y) <= Atom.length(X, Y, s)[1,] for s in [Term.short, Term.long]],
            *[Atom.sides(Y) <= Atom.sides(X, Y, s)[1,] for s in [Term.not_double, Term.double]],
            *[Atom.roof(Y) <= Atom.roof(X, Y, s)[1,] for s in roofs],
            *[Atom.wheels(Y) <= Atom.wheels(X, Y, s)[1,] for s in [2, 3]],
            *[Atom.loadnum(Y) <= Atom.loadnum(X, Y, s)[1,] for s in [0, 1, 2, 3]],
            *[Atom.loadshape(Y) <= Atom.loadshape(X, Y, s)[1,] for s in loadshapes],
            Atom.vagon(Y) <= (atom(Y)[1,] for atom in vagon_atoms),
            Atom.train <= Atom.vagon(Y)[1,],
            Atom.direction <= Atom.train[1,],
        ]
    )

    examples: List[List] = [[]] * 20

    for _, id, pos, shape, length, sides, roof, wheels, load, loadnum in train_example_data:
        if not examples[id - 1]:
            examples[id - 1] = []
        examples[id - 1].extend(
            [
                Atom.shape(id, pos, shape),
                Atom.length(id, pos, length),
                Atom.sides(id, pos, sides),
                Atom.roof(id, pos, roof),
                Atom.wheels(id, pos, wheels),
                Atom.loadshape(id, pos, load),
                Atom.loadnum(id, pos, loadnum),
            ]
        )

    model.add_examples(examples)

    model.add_queries([*[Atom.direction[1.0] for _ in range(1, 11)], *[Atom.direction[-1.0] for _ in range(11, 21)]])

    dataset = model.build()  # Build model into dataset (weights and samples)
