from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation
from neuralogic.core.constructs.factories import R
from neuralogic.nn.module.module import Module


class Linear(Module):
    r"""
    Apply linear transformation on the input. Can be expressed as:

    .. math::

        h_{i_0, .., i_{n}} = W \cdot x_{i_0, .., i_{n}}

    Where :math:`x` is the input, :math:`W \in R^{(out\_channels \times in\_channels)}` is a learnable parameter,
    and :math:`n` is the arity of the input and output.

    It is also possible to attach non-linearity via the activation parameter and compute:

    .. math::

        h_{i_0, .., i_{n}} = act(W \cdot x_{i_0, .., i_{n}})

    Example
    -------

    The whole computation of this module (parametrized as :code:`Linear(1, 2, "h1", "h0")`) is as follows:

    .. code:: logtalk

        (R.h1(V.X0)[2, 1] <= R.h0(V.X0)) | [Transformation.IDENTITY]
        R.h1 / 1 | [Transformation.IDENTITY]

    Module parametrized as :code:`Linear(1, 2, "h1", "h0", Transformation.SIGMOID, 2)` translates into:

    .. code:: logtalk

        (R.h1(V.X0, V.X1)[2, 1] <= R.h0(V.X0, V.X1)) | [Transformation.IDENTITY]
        R.h1 / 2 | [Transformation.SIGMOID]

    Parameters
    ----------

    in_channels : int
        Input feature size.
    out_channels : int
        Output feature size.
    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input name.
    activation : Transformation
        Activation function of the output.
        Default: ``Transformation.IDENTITY``
    arity : int
        Arity of the input and output predicate. Default: ``1``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        input_name: str,
        activation: Transformation = Transformation.IDENTITY,
        arity: int = 1,
    ):
        self.output_name = output_name
        self.input_name = input_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation = activation
        self.arity = arity

    def __call__(self):
        terms = [f"X{i}" for i in range(self.arity)]
        head = R.get(self.output_name)(terms)[self.out_channels, self.in_channels]

        return [
            (head <= R.get(self.input_name)(terms)) | [Transformation.IDENTITY],
            R.get(self.output_name) / len(terms) | Metadata(transformation=self.activation),
        ]
