from neuralogic.core import Atom, Template, Var
from neuralogic.utils.templates.component import AbstractComponent


class GIN(AbstractComponent):
    def build(self, template: Template, layer_count: int, previous_name: str) -> str:
        name = f"l{layer_count}_gin"
        embed_name = f"l{layer_count}_gin_embed"
        previous_name = previous_name or AbstractComponent.features_name

        head_atom = Atom.get(embed_name)(Var.X)
        template.add_rule(head_atom <= (Atom.get(previous_name)(Var.Y), Atom.edge(Var.X, Var.Y)))
        template.add_rule(head_atom <= Atom.get(previous_name)(Var.X))
        template.add_rule(Atom.get(name)(Var.X)[self.weight_shape] <= head_atom[self.weight_shape])

        return name
