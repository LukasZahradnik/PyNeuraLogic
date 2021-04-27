from typing import List
from examples.datasets.data.train_example_data import train_example_data

from neuralogic.core import Atom, Problem, Var, Term
from neuralogic.core.settings import Settings, Optimizer

settings = Settings(optimizer=Optimizer.SGD, epochs=300)


with Problem(settings).context() as problem:
    # One example per train, doesn't know order of vagons

    # fmt: off

    shapes = [Term.ellipse, Term.rectangle, Term.bucket, Term.hexagon, Term.u_shaped]
    roofs = [Term.jagged, Term.arc, Term.none, Term.flat, Term.peaked]
    loadshapes = [Term.hexagon, Term.triangle, Term.diamond, Term.rectangle, Term.circle]
    vagon_atoms = [Atom.shape, Atom.length, Atom.sides, Atom.wheels, Atom.loadnum, Atom.loadshape, Atom.roof]

    Y = Var.Y

    problem.add_rules(
        [
            *[Atom.shape(Y) <= Atom.shape(Y, s)[1,] for s in shapes],
            *[Atom.length(Y) <= Atom.length(Y, s)[1,] for s in [Term.short, Term.long]],
            *[Atom.sides(Y) <= Atom.sides(Y, s)[1,] for s in [Term.not_double, Term.double]],
            *[Atom.roof(Y) <= Atom.roof(Y, s)[1,] for s in roofs],
            *[Atom.wheels(Y) <= Atom.wheels(Y, s)[1,] for s in [2, 3]],
            *[Atom.loadnum(Y) <= Atom.loadnum(Y, s)[1,] for s in [0, 1, 2, 3]],
            *[Atom.loadshape(Y) <= Atom.loadshape(Y, s)[1,] for s in loadshapes],
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
                Atom.shape(pos, shape),
                Atom.length(pos, length),
                Atom.sides(pos, sides),
                Atom.roof(pos, roof),
                Atom.wheels(pos, wheels),
                Atom.loadshape(pos, load),
                Atom.loadnum(pos, loadnum),
            ]
        )

    problem.add_examples(examples)

    problem.add_queries([*[Atom.direction[1.0] for _ in range(1, 11)], *[Atom.direction[-1.0] for _ in range(11, 21)]])
