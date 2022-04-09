from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class SimplE(Module):
    r"""

    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    head_entity_embed_name : str
    tail_entity_embed_name : str
    relation_embed_name : str
    inv_relation_embed_name : str
    activation : Activation
        Activation function of the output.
        Default: ``Activation.IDENTITY``
    """

    def __init__(
        self,
        output_name: str,
        head_entity_embed_name: str,
        tail_entity_embed_name: str,
        relation_embed_name: str,
        inv_relation_embed_name: str,
        activation: Activation = Activation.IDENTITY,
    ):
        self.output_name = output_name
        self.relation_embed_name = relation_embed_name
        self.inv_relation_embed_name = inv_relation_embed_name
        self.head_entity_embed_name = head_entity_embed_name
        self.tail_entity_embed_name = tail_entity_embed_name

        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.H, V.R, V.T)
        h_ent_embed = R.get(self.head_entity_embed_name)
        t_ent_embed = R.get(self.tail_entity_embed_name)

        inv_rel_embed = R.get(self.inv_relation_embed_name)
        rel_embed = R.get(self.relation_embed_name)

        metadata = Metadata(activation="product-identity")

        return [
            (
                R.get(f"{self.output_name}__simple")(V.H, V.R, V.T) <= (
                    h_ent_embed(V.T), inv_rel_embed(V.R), t_ent_embed(V.H)
                )
            ) | metadata,
            (
                R.get(f"{self.output_name}__simple")(V.H, V.R, V.T) <= (
                    h_ent_embed(V.H), rel_embed(V.R), t_ent_embed(V.T)
                )
            ) | metadata,
            R.get(f"{self.output_name}__simple") / 3 | [Activation.IDENTITY, Aggregation.SUM],

            head[0.5].fixed() <= R.get(f"{self.output_name}__simple")(V.H, V.R, V.T) [Activation.IDENTITY],
            R.get(self.output_name) / 1 | Metadata(activation=self.activation),
        ]
