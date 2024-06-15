from neuralogic.core import Relation, Template, Var, Const
from neuralogic.dataset import Dataset


dataset = Dataset()
template = Template()

# fmt: off

template.add_rules(
    [
        Relation.foal(Var.X)[1, ] <= (Relation.parent(Var.X, Var.Y), Relation.horse(Var.Y)),  # todo gusta: mozna prejmenovat Atom -> Predicate by odpovidalo skutecnosti prirozeneji?
        Relation.foal(Var.X)[1, ] <= (Relation.sibling(Var.X, Var.Y), Relation.horse(Var.Y)),
        Relation.negFoal(Var.X)[1, ] <= Relation.foal(Var.X),
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
