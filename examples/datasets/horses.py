from neuralogic.core import Atom, Template, Var, Term
from neuralogic.core.settings import Settings, Optimizer
from neuralogic.utils.data import Dataset


settings = Settings(optimizer=Optimizer.SGD, epochs=300)
dataset = Dataset()

with Template(settings).context() as template:
    # fmt: off

    template.add_rules(
        [
            Atom.foal(Var.X)[1,] <= (Atom.parent(Var.X, Var.Y), Atom.horse(Var.Y)),
            Atom.foal(Var.X)[1,] <= (Atom.sibling(Var.X, Var.Y), Atom.horse(Var.Y)),
            Atom.negFoal(Var.X)[1,] <= Atom.foal(Var.X),
        ]
    )

    dataset.add_example(
        [
            Atom.horse(Term.aida)[1.0],
            Atom.horse(Term.cheyenne)[1.0],
            Atom.horse(Term.dakotta)[1.0],
            Atom.parent(Term.star, Term.cheyenne)[1.0],
            Atom.parent(Term.star, Term.aida)[1.0],
            Atom.parent(Term.star, Term.dakotta)[1.0],
        ]
    )

    dataset.add_queries(
        [
            Atom.foal(Term.star)[1.0],
            Atom.negFoal(Term.star)[0.0],
        ]
    )
