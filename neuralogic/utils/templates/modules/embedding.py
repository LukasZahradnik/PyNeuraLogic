from typing import List

from neuralogic.core import Relation, Template, Var, Activation, Aggregation, Metadata
from neuralogic.utils.templates.modules import AbstractModule


class Embedding(AbstractModule):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        name=None,
    ):
        super().__init__(
            name=name,
            in_channels=-1,
            out_channels=-1,
        )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def build(
        self, template: Template, layer_count: int, previous_names: List[str], feature_name: str, edge_name: str
    ) -> str:
        name = f"l{layer_count}_embedding" if self.name is None else self.name
        previous_name = feature_name if len(previous_names) == 0 else previous_names[-1]

        head_atom = Relation.get(name)(Var.X)
        feature_rule = head_atom[self.num_embeddings, self.embedding_dim] <= Relation.get(previous_name)(Var.X)

        #todo gusta: toto je jenom linearni projekci vrstva, tj. embedding jen pokud maji i vstupy unikatni (one-hot) hodnoty.
        # Nova verze pro unikatni embeddings je:
        # @embed...(X,...) :- cokoliv(X,...),...
        # ..to vytvori skutecne unikatni @embed...(x) Valued Atom za kazdy validni unikatni objekt x

        template.add_rule(feature_rule | Metadata(activation=Activation.IDENTITY))
        template.add_rule(Relation.get(name) / 1 | Metadata(activation=Activation.IDENTITY))

        return name