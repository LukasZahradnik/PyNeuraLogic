from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.constructs.function import Transformation, Aggregation
from neuralogic.core.constructs.factories import R
from neuralogic.nn.module.module import Module


class Pooling(Module):
    r"""
    Apply generic pooling over the input specified by the input_name and the input arity parameters.
    Can be expressed as:

    .. math::
        h = agg_{i_{0}, .., i_{n} \in N}(x_{(i_{0}, .., i_{n})})

    Where :math:`N` is a set of tuples of length :math:`n` (specified by the input arity parameter)
    that are valid arguments for the input predicate.

    For example, a classic pooling over graph nodes represented by relations of arity 1 (node id)
    would be calculated as:

    .. math::
        h = agg_{i \in N}(x_{(i)})

    Here :math:`N` refers to a set of all node ids. Lifting the restriction of the input arity via the input_arity
    parameter allows for pooling not only nodes but also edges (``input_arity=2``) and other objects (hyperedges etc.)

    Examples
    --------

    The whole computation of this module (parametrized as :code:`Pooling("h1", "h0", Aggregation.AVG)`) is as follows:

    .. code:: logtalk

        (R.h1 <= R.h0(V.X0)) | [Aggregation.AVG, Transformation.IDENTITY]
        R.h1 / 0 | [Transformation.IDENTITY]

    Module parametrized as :code:`Pooling("h1", "h0", Aggregation.MAX, 2)` translates into:

    .. code:: logtalk

        (R.h1 <= R.h0(V.X0, V.X1)) | [Aggregation.MAX, Transformation.IDENTITY]
        R.h1 / 0 | [Transformation.IDENTITY]

    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input name.
    aggregation : Aggregation
        Aggregation function.
    input_arity : int
        Arity of the input predicate ``input_name``. Default: ``1``
    """

    def __init__(
        self,
        output_name: str,
        input_name: str,
        aggregation: Aggregation,
        input_arity: int = 1,
    ):
        self.output_name = output_name
        self.input_name = input_name
        self.input_arity = input_arity

        self.aggregation = aggregation

    def __call__(self):
        metadata = Metadata(transformation=Transformation.IDENTITY, aggregation=self.aggregation)

        return [
            (R.get(self.output_name) <= R.get(self.input_name)(f"X{i}" for i in range(self.input_arity))) | metadata,
            R.get(self.output_name) / 0 | [Transformation.IDENTITY],
        ]


class MaxPooling(Pooling):
    r"""
    Apply max pooling over the input specified by the input_name and the input arity parameters.
    Can be expressed as:

    .. math::
        h = max_{i_{0}, .., i_{n} \in N}(x_{(i_{0}, .., i_{n})})

    Where :math:`N` is a set of tuples of length :math:`n` (specified by the input arity parameter)
    that are valid arguments for the input predicate.

    This module extends the generic pooling :class:`~neuralogic.nn.module.pooling.Pooling`.

    Examples
    --------

    The whole computation of this module (parametrized as :code:`MaxPooling("h1", "h0")`) is as follows:

    .. code:: logtalk

        (R.h1 <= R.h0(V.X0)) | [Aggregation.MAX, Transformation.IDENTITY]
        R.h1 / 0 | [Transformation.IDENTITY]

    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input name.
    input_arity : int
        Arity of the input predicate ``input_name``. Default: ``1``
    """

    def __init__(self, output_name: str, input_name: str, input_arity: int = 1):
        super().__init__(output_name, input_name, Aggregation.MAX, input_arity)


class AvgPooling(Pooling):
    r"""
    Apply average pooling over the input specified by the input_name and the input arity parameters.
    Can be expressed as:

    .. math::
        h = \frac{1}{|N|}\sum_{i_{0}, .., i_{n} \in N} x_{(i_{0}, .., i_{n})}

    Where :math:`N` is a set of tuples of length :math:`n` (specified by the input arity parameter)
    that are valid arguments for the input predicate.

    This module extends the generic pooling :class:`~neuralogic.nn.module.pooling.Pooling`.

    Examples
    --------

    The whole computation of this module (parametrized as :code:`AvgPooling("h1", "h0")`) is as follows:

    .. code:: logtalk

        (R.h1 <= R.h0(V.X0)) | [Aggregation.AVG, Transformation.IDENTITY]
        R.h1 / 0 | [Transformation.IDENTITY]

    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input name.
    input_arity : int
        Arity of the input predicate ``input_name``. Default: ``1``
    """

    def __init__(self, output_name: str, input_name: str, input_arity: int = 1):
        super().__init__(output_name, input_name, Aggregation.AVG, input_arity)


class SumPooling(Pooling):
    r"""
    Apply sum pooling over the input specified by the input_name and the input arity parameters.
    Can be expressed as:

    .. math::
        h = \sum_{i_{0}, .., i_{n} \in N} x_{(i_{0}, .., i_{n})}

    Where :math:`N` is a set of tuples of length :math:`n` (specified by the input arity parameter)
    that are valid arguments for the input predicate.

    This module extends the generic pooling :class:`~neuralogic.nn.module.pooling.Pooling`.

    Examples
    --------

    The whole computation of this module (parametrized as :code:`SumPooling("h1", "h0")`) is as follows:

    .. code:: logtalk

        (R.h1 <= R.h0(V.X0)) | [Aggregation.SUM, Transformation.IDENTITY]
        R.h1 / 0 | [Transformation.IDENTITY]

    Parameters
    ----------

    output_name : str
        Output (head) predicate name of the module.
    input_name : str
        Input name.
    input_arity : int
        Arity of the input predicate ``input_name``. Default: ``1``
    """

    def __init__(self, output_name: str, input_name: str, input_arity: int = 1):
        super().__init__(output_name, input_name, Aggregation.SUM, input_arity)
