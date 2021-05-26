from typing import List

from neuralogic.core import Atom, Template, Var, Activation, Aggregation, Metadata
from neuralogic.utils.templates.modules import AbstractModule


class Embedding(AbstractModule):
    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        name=None,
    ):
        super().__init__(
            name=name,
            in_channels=-1,
            out_channels=-1,
        )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def build(self, template: Template, layer_count: int, previous_names: List[str]) -> str:
        name = f"l{layer_count}_embedding" if self.name is None else self.name
        previous_name = AbstractModule.features_name if len(previous_names) == 0 else previous_names[-1]

        head_atom = Atom.get(name)(Var.X)
        feature_rule = head_atom[self.num_embeddings, self.embedding_dim] <= Atom.get(previous_name)(Var.X)

        template.add_rule(feature_rule | Metadata(activation=Activation.IDENTITY))
        template.add_rule(Atom.get(name) / 1 | Metadata(activation=Activation.IDENTITY))

        return name
