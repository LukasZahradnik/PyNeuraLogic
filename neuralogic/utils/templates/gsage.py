from neuralogic.core import Atom, Template, Var
from neuralogic.utils.templates.component import AbstractComponent


class GraphSAGE(AbstractComponent):
    def build(self, template: Template, layer_count: int, previous_name: str):
        name = f"l{layer_count}_gsage"
        previous_name = previous_name or AbstractComponent.features_name

        head_atom = Atom.get(name)(Var.X)
        template.add_rule(head_atom[self.weight_shape] <= (Atom.get(previous_name)(Var.Y), Atom.edge(Var.X, Var.Y)))
        template.add_rule(head_atom[self.weight_shape] <= Atom.get(previous_name)(Var.X))
        return name
