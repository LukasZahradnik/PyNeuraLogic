from examples.datasets.data.train_example_data import train_example_data

from neuralogic.core import Relation, Template, Var, Constant
from neuralogic.dataset import Dataset


dataset = Dataset()
template = Template()

# Naive trains - one big example

# fmt: off

shapes = [Constant.ellipse, Constant.rectangle, Constant.bucket, Constant.hexagon, Constant.u_shaped]
roofs = [Constant.jagged, Constant.arc, Constant.none, Constant.flat, Constant.peaked]
loadshapes = [Constant.hexagon, Constant.triangle, Constant.diamond, Constant.rectangle, Constant.circle]
vagon_atoms = [Relation.shape, Relation.length, Relation.sides, Relation.wheels, Relation.loadnum, Relation.loadshape, Relation.roof]

X = Var.X
Y = Var.Y

template.add_rules(
    [
        *[Relation.shape(X, Y) <= Relation.shape(X, Y, s)[1,] for s in shapes],
        *[Relation.length(X, Y) <= Relation.length(X, Y, s)[1,] for s in [Constant.short, Constant.long]],
        *[Relation.sides(X, Y) <= Relation.sides(X, Y, s)[1,] for s in [Constant.not_double, Constant.double]],
        *[Relation.roof(X, Y) <= Relation.roof(X, Y, s)[1,] for s in roofs],
        *[Relation.wheels(X, Y) <= Relation.wheels(X, Y, s)[1,] for s in [2, 3]],
        *[Relation.loadnum(X, Y) <= Relation.loadnum(X, Y, s)[1,] for s in [0, 1, 2, 3]],
        *[Relation.loadshape(X, Y) <= Relation.loadshape(X, Y, s)[1,] for s in loadshapes],
        Relation.vagon(X, Y) <= (atom(X, Y)[1,] for atom in vagon_atoms),
        *[Relation.train(X) <= Relation.vagon(X, i)[1,] for i in [1, 2, 3, 4]],
        Relation.direction(X) <= Relation.train(X)[1,],
    ]
)

dataset.add_example(
    [
        atom
        for _, id, pos, shape, length, sides, roof, wheels, load, loadnum in train_example_data
        for atom in [
            Relation.shape(id, pos, shape),
            Relation.length(id, pos, length),
            Relation.sides(id, pos, sides),
            Relation.roof(id, pos, roof),
            Relation.wheels(id, pos, wheels),
            Relation.loadshape(id, pos, load),
            Relation.loadnum(id, pos, loadnum),
        ]
    ]
)

dataset.add_queries(
    [*[Relation.direction(i)[1.0] for i in range(1, 11)], *[Relation.direction(i)[-1.0] for i in range(11, 21)]]
)
