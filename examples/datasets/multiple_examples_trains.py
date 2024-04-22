from typing import List
from examples.datasets.data.train_example_data import train_example_data

from neuralogic.core import Relation, Template, Var, Constant, Transformation
from neuralogic.dataset import Dataset


dataset = Dataset()
template = Template()

# One example per train

# fmt: off

shapes = [Constant.ellipse, Constant.rectangle, Constant.bucket, Constant.hexagon, Constant.u_shaped]
roofs = [Constant.jagged, Constant.arc, Constant.none, Constant.flat, Constant.peaked]
loadshapes = [Constant.hexagon, Constant.triangle, Constant.diamond, Constant.rectangle, Constant.circle]
vagon_atoms = [Relation.shape, Relation.length, Relation.sides, Relation.wheels, Relation.loadnum, Relation.loadshape, Relation.roof]

Y = Var.Y

meta = [Transformation.TANH]

template.add_rules(
    [
        *[(Relation.shape(Y) <= Relation.shape(Y, s)[1, ]) | meta for s in shapes],
        *[(Relation.length(Y) <= Relation.length(Y, s)[1, ]) | meta for s in [Constant.short, Constant.long]],
        *[(Relation.sides(Y) <= Relation.sides(Y, s)[1, ]) | meta for s in [Constant.not_double, Constant.double]],
        *[(Relation.roof(Y) <= Relation.roof(Y, s)[1, ]) | meta for s in roofs],
        *[(Relation.wheels(Y) <= Relation.wheels(Y, s)[1, ]) | meta for s in [2, 3]],
        *[(Relation.loadnum(Y) <= Relation.loadnum(Y, s)[1, ]) | meta for s in [0, 1, 2, 3]],
        *[(Relation.loadshape(Y) <= Relation.loadshape(Y, s)[1, ]) | meta for s in loadshapes],
        (Relation.vagon(Y) <= (atom(Y)[1, ] for atom in vagon_atoms)) | meta,
        *[(Relation.train <= Relation.vagon(i)[1, ]) | meta for i in [1, 2, 3, 4]],
        (Relation.direction <= Relation.train[1, ]) | meta,
        Relation.shape / 1 | meta,
        Relation.length / 1 | meta,
        Relation.sides / 1 | meta,
        Relation.roof / 1 | meta,
        Relation.wheels / 1 | meta,
        Relation.loadnum / 1 | meta,
        Relation.loadshape / 1 | meta,
        Relation.vagon / 1 | meta,
        Relation.train / 0 | meta,
        Relation.direction / 0 | meta,
    ]
)

examples: List[List] = [[]] * 20
for _, id, pos, shape, length, sides, roof, wheels, load, loadnum in train_example_data:
    if not examples[id - 1]:
        examples[id - 1] = []
    examples[id - 1].extend(
        [
            Relation.shape(pos, shape),
            Relation.length(pos, length),
            Relation.sides(pos, sides),
            Relation.roof(pos, roof),
            Relation.wheels(pos, wheels),
            Relation.loadshape(pos, load),
            Relation.loadnum(pos, loadnum),
        ]
    )


for example in examples[:10]:
    dataset.add(Relation.direction[1.0], example)

for example in examples[10:]:
    dataset.add(Relation.direction[-1.0], example)
