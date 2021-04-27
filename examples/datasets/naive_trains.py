from examples.datasets.data.train_example_data import train_example_data

from neuralogic.core import Atom, Problem, Var, Term
from neuralogic.core.settings import Settings, Optimizer

settings = Settings(optimizer=Optimizer.SGD, epochs=300)


with Problem(settings).context() as problem:
    # Naive trains - one big example

    # fmt: off

    shapes = [Term.ellipse, Term.rectangle, Term.bucket, Term.hexagon, Term.u_shaped]
    roofs = [Term.jagged, Term.arc, Term.none, Term.flat, Term.peaked]
    loadshapes = [Term.hexagon, Term.triangle, Term.diamond, Term.rectangle, Term.circle]
    vagon_atoms = [Atom.shape, Atom.length, Atom.sides, Atom.wheels, Atom.loadnum, Atom.loadshape, Atom.roof]

    X = Var.X
    Y = Var.Y

    problem.add_rules(
        [
            *[Atom.shape(X, Y) <= Atom.shape(X, Y, s)[1,] for s in shapes],
            *[Atom.length(X, Y) <= Atom.length(X, Y, s)[1,] for s in [Term.short, Term.long]],
            *[Atom.sides(X, Y) <= Atom.sides(X, Y, s)[1,] for s in [Term.not_double, Term.double]],
            *[Atom.roof(X, Y) <= Atom.roof(X, Y, s)[1,] for s in roofs],
            *[Atom.wheels(X, Y) <= Atom.wheels(X, Y, s)[1,] for s in [2, 3]],
            *[Atom.loadnum(X, Y) <= Atom.loadnum(X, Y, s)[1,] for s in [0, 1, 2, 3]],
            *[Atom.loadshape(X, Y) <= Atom.loadshape(X, Y, s)[1,] for s in loadshapes],
            Atom.vagon(X, Y) <= (atom(X, Y)[1,] for atom in vagon_atoms),
            *[Atom.train(X) <= Atom.vagon(X, i)[1,] for i in [1, 2, 3, 4]],
            Atom.direction(X) <= Atom.train(X)[1,],
        ]
    )

    problem.add_example(
        [
            atom
            for _, id, pos, shape, length, sides, roof, wheels, load, loadnum in train_example_data
            for atom in [
                Atom.shape(id, pos, shape),
                Atom.length(id, pos, length),
                Atom.sides(id, pos, sides),
                Atom.roof(id, pos, roof),
                Atom.wheels(id, pos, wheels),
                Atom.loadshape(id, pos, load),
                Atom.loadnum(id, pos, loadnum),
            ]
        ]
    )

    problem.add_queries(
        [*[Atom.direction(i)[1.0] for i in range(1, 11)], *[Atom.direction(i)[-1.0] for i in range(11, 21)]]
    )
