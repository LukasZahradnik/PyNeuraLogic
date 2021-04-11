from neuralogic.model import Atom, Model, Var, Term

with Model().context() as model:
    # fmt: off

    model.add_rules(
        [
            Atom.foal(Var.X)[1,] <= (Atom.parent(Var.X, Var.Y), Atom.horse(Var.Y)),
            Atom.foal(Var.X)[1,] <= (Atom.sibling(Var.X, Var.Y), Atom.horse(Var.Y)),
            Atom.negFoal(Var.X)[1,] <= Atom.foal(Var.X),
        ]
    )

    model.add_example(
        [
            Atom.horse(Term.aida)[1.0],
            Atom.horse(Term.cheyenne)[1.0],
            Atom.horse(Term.dakotta)[1.0],
            Atom.parent(Term.star, Term.cheyenne)[1.0],
            Atom.parent(Term.star, Term.aida)[1.0],
            Atom.parent(Term.star, Term.dakotta)[1.0],
        ]
    )

    model.add_queries(
        [
            Atom.foal(Term.star)[1.0],
            Atom.negFoal(Term.star)[0.0],
        ]
    )

    dataset = model.build()  # Build model into dataset (weights and samples)
