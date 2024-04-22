from neuralogic.core import Relation, Template, Var, Constant, Transformation
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
    Relation.horse(Constant.aida)[1.0],
    Relation.horse(Constant.cheyenne)[1.0],
    Relation.horse(Constant.dakotta)[1.0],
    Relation.parent(Constant.star, Constant.cheyenne)[1.0],
    Relation.parent(Constant.star, Constant.aida)[1.0],
    Relation.parent(Constant.star, Constant.dakotta)[1.0],
]

dataset.add(Relation.foal(Constant.star)[1.0], example)
dataset.add(Relation.negFoal(Constant.star)[0.0], example)
