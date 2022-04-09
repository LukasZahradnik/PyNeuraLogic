from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class DistMult(Module):
    r"""

    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    relation_embed_name : str
    entity_embed_name : str
    activation : Activation
        Activation function of the output.
        Default: ``Activation.IDENTITY``

    """

    def __init__(
        self,
        output_name: str,
        relation_embed_name: str,
        entity_embed_name: str,
        activation: Activation = Activation.IDENTITY,
    ):
        self.output_name = output_name
        self.relation_embed_name = relation_embed_name
        self.entity_embed_name = entity_embed_name

        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I, V.R, V.J)
        ent_embed = R.get(self.entity_embed_name)
        rel_embed = R.get(self.relation_embed_name)

        metadata = Metadata(activation="product-identity", aggregation=Aggregation.SUM)

        return [
            (head[-1].fixed() <= (ent_embed(V.I), rel_embed(V.R), ent_embed(V.J))) | metadata,
            R.get(self.output_name) / 1 | Metadata(activation=self.activation),
        ]
