from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.nn.module.module import Module


class RvNN(Module):
    r"""
    Recursive Neural Network (RvNN) module which is computed as:

    .. math::

         \mathbf{h}_i = act(agg_{j \in \mathcal{Ch(i)}}(\mathbf{W_{id(j)}} \mathbf{h}_j))

    Where :math:`act` is an activation function, :math:`agg` aggregation function and :math:`\mathbf{W}`'s
    are learnable parameters. :math:`\mathcal{Ch(i)}` represents the ordered list of children of node :math:`i`.
    The :math:`id(j)` function maps node :math:`j` to its index (position) in its parent's children list.

    Parameters
    ----------

    input_size : int
        Input feature size.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input feature predicate name to get leaf features from.
    parent_map_name : str
        Name of the predicate to get mapping from parent to children
    max_children : int
        Maximum number of children (specify which <max_children>-ary tree will be considered).
        Default: ``2``
    activation : Transformation
        Activation function of all layers.
        Default: ``Transformation.TANH``
    aggregation : Aggregation
        Aggregation function of a layer.
        Default: ``Aggregation.SUM``
    arity : int
        Arity of the input and output predicate (doesn't include the node id term). Default: ``1``
    """

    def __init__(
        self,
        input_size: int,
        output_name: str,
        input_name: str,
        parent_map_name: str,
        max_children: int = 2,
        activation: Transformation = Transformation.TANH,
        aggregation: Aggregation = Aggregation.SUM,
        arity: int = 1,
    ):
        self.input_size = input_size

        self.output_name = output_name
        self.input_name = input_name
        self.parent_map_name = parent_map_name
        self.max_children = max_children

        self.activation = activation
        self.aggregation = aggregation
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        head_terms = [*terms, V.P]

        input_rel = R.get(self.input_name)
        output_rel = R.get(self.output_name)
        parent_map_rel = R.get(self.parent_map_name)
        metadata = Metadata(transformation=self.activation, aggregation=self.aggregation)

        rules = [
            (output_rel(head_terms) <= (input_rel(head_terms), parent_map_rel(V.P))) | metadata,
            output_rel / len(head_terms) | [Transformation.IDENTITY],
        ]

        body = []
        parent_terms = [V.P]

        for i in range(1, self.max_children + 1):
            term = f"C{i}"
            body.append(output_rel([*terms, term])[f"{self.output_name}__rvnn_{i}" : self.input_size, self.input_size])
            parent_terms.append(term)

            rules.append((output_rel(head_terms) <= (*body, parent_map_rel(*parent_terms))) | metadata)
        return rules
