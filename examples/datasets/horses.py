from neuralogic.core import Relation, Template, Var, Constant
from neuralogic.dataset import Dataset


dataset = Dataset()
template = Template()

# fmt: off

template.add_rules(
    [
        Relation.foal(Var.X)[1,] <= (Relation.parent(Var.X, Var.Y), Relation.horse(Var.Y)), # todo gusta: mozna prejmenovat Atom -> Predicate by odpovidalo skutecnosti prirozeneji?
        Relation.foal(Var.X)[1,] <= (Relation.sibling(Var.X, Var.Y), Relation.horse(Var.Y)),
        Relation.negFoal(Var.X)[1,] <= Relation.foal(Var.X),
    ]
)

dataset.add_example(
    [
        Relation.horse(Constant.aida)[1.0],
        Relation.horse(Constant.cheyenne)[1.0],
        Relation.horse(Constant.dakotta)[1.0],
        Relation.parent(Constant.star, Constant.cheyenne)[1.0],
        Relation.parent(Constant.star, Constant.aida)[1.0],
        Relation.parent(Constant.star, Constant.dakotta)[1.0],
    ]
)

dataset.add_queries(
    [
        Relation.foal(Constant.star)[1.0],
        Relation.negFoal(Constant.star)[0.0],
    ]
)
