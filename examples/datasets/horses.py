from neuralogic.core import Relation, Template, Var, Const, Transformation
from neuralogic.dataset import Dataset


dataset = Dataset()
template = Template()

# fmt: off

template.add_rules(
    [
        (Relation.foal(Var.X)[1, ] <= (Relation.parent(Var.X, Var.Y), Relation.horse(Var.Y))) | [Transformation.TANH],
        (Relation.foal(Var.X)[1, ] <= (Relation.sibling(Var.X, Var.Y), Relation.horse(Var.Y))) | [Transformation.TANH],
        (Relation.negFoal(Var.X)[1, ] <= Relation.foal(Var.X)) | [Transformation.TANH],
        Relation.foal / 1 | [Transformation.TANH],
        Relation.negFoal / 1 | [Transformation.TANH],
    ]
)

example = [
    Relation.horse(Const.aida)[1.0],
    Relation.horse(Const.cheyenne)[1.0],
    Relation.horse(Const.dakotta)[1.0],
    Relation.parent(Const.star, Const.cheyenne)[1.0],
    Relation.parent(Const.star, Const.aida)[1.0],
    Relation.parent(Const.star, Const.dakotta)[1.0],
]

dataset.add(Relation.foal(Const.star)[1.0], example)
dataset.add(Relation.negFoal(Const.star)[0.0], example)
