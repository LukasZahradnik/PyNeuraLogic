from examples.datasets.data.train_example_data import train_example_data

from neuralogic.core import Relation, Template, Var, Const, Transformation
from neuralogic.dataset import Dataset


dataset = Dataset()
template = Template()

# Naive trains - one big example

# fmt: off

shapes = [Const.ellipse, Const.rectangle, Const.bucket, Const.hexagon, Const.u_shaped]
roofs = [Const.jagged, Const.arc, Const.none, Const.flat, Const.peaked]
loadshapes = [Const.hexagon, Const.triangle, Const.diamond, Const.rectangle, Const.circle]
vagon_atoms = [Relation.shape, Relation.length, Relation.sides, Relation.wheels, Relation.loadnum, Relation.loadshape, Relation.roof]

X = Var.X
Y = Var.Y

meta = [Transformation.TANH]

template.add_rules(
    [
        *[(Relation.shape(X, Y) <= Relation.shape(X, Y, s)[1, ]) | meta for s in shapes],
        *[(Relation.length(X, Y) <= Relation.length(X, Y, s)[1, ]) | meta for s in [Const.short, Const.long]],
        *[(Relation.sides(X, Y) <= Relation.sides(X, Y, s)[1, ]) | meta for s in [Const.not_double, Const.double]],
        *[(Relation.roof(X, Y) <= Relation.roof(X, Y, s)[1, ]) | meta for s in roofs],
        *[(Relation.wheels(X, Y) <= Relation.wheels(X, Y, s)[1, ]) | meta for s in [2, 3]],
        *[(Relation.loadnum(X, Y) <= Relation.loadnum(X, Y, s)[1, ]) | meta for s in [0, 1, 2, 3]],
        *[(Relation.loadshape(X, Y) <= Relation.loadshape(X, Y, s)[1, ]) | meta for s in loadshapes],
        (Relation.vagon(X, Y) <= (atom(X, Y)[1, ] for atom in vagon_atoms)) | meta,
        *[(Relation.train(X) <= Relation.vagon(X, i)[1, ]) | meta for i in [1, 2, 3, 4]],
        (Relation.direction(X) <= Relation.train(X)[1, ]) | meta,
        Relation.shape / 2 | meta,
        Relation.length / 2 | meta,
        Relation.sides / 2 | meta,
        Relation.roof / 2 | meta,
        Relation.wheels / 2 | meta,
        Relation.loadnum / 2 | meta,
        Relation.loadshape / 2 | meta,
        Relation.vagon / 2 | meta,
        Relation.train / 1 | meta,
        Relation.direction / 1 | meta,
    ]
)

example = [
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

for i in range(1, 11):
    dataset.add(Relation.direction(i)[1.0], example)

for i in range(11, 21):
    dataset.add(Relation.direction(i)[-1.0], example)
