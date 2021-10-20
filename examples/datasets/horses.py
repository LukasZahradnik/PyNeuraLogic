from neuralogic.core import Relation, Template, Var, Term
from neuralogic.core.settings import Settings, Optimizer
from neuralogic.utils.data import Dataset


settings = Settings(optimizer=Optimizer.SGD, epochs=300)
dataset = Dataset()

with Template(settings).context() as template:
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
            Relation.horse(Term.aida)[1.0],
            Relation.horse(Term.cheyenne)[1.0],
            Relation.horse(Term.dakotta)[1.0],
            Relation.parent(Term.star, Term.cheyenne)[1.0],
            Relation.parent(Term.star, Term.aida)[1.0],
            Relation.parent(Term.star, Term.dakotta)[1.0],
        ]
    )

    dataset.add_queries(
        [
            Relation.foal(Term.star)[1.0],
            Relation.negFoal(Term.star)[0.0],
        ]
    )
