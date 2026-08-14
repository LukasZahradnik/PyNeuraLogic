class ErrorFunctionNames:
    MSE = "SQUARED_DIFF"
    CROSSENTROPY = "CROSSENTROPY"
    SOFTENTROPY = "SOFTENTROPY"


class ErrorFunction:
    """
    Base class for error (loss) functions in the neural network.

    Every one of them takes a ``reduction``, as in torch, and it means the same thing: how a batch's
    per-query errors are reduced into the single quantity being descended, and therefore what the gradient
    is. ``"mean"`` divides by the total element count and ``"sum"`` does not divide at all.

    The divisor generalises torch's ``N x C`` to the two things a relational engine has and torch does not -
    a query's ``importance``, and a target whose width can differ between queries - as the sum of importance
    times width over the batch. With unit importances and equal widths that is exactly ``N x C``.

    Note the per-query values that ``validate()`` and ``train()`` hand back are *not* reduced across the
    batch: they are the un-aggregated errors, each summed over its own components, which is torch's
    ``reduction="none"``. The reduction governs the gradient and any aggregate over the batch.
    """

    def __init__(self, reduction: str = "mean"):
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.reduction = reduction


class MSE(ErrorFunction):
    """
    Mean Squared Error (SQUARED_DIFF) loss function.
    Suitable for regression tasks.

    Parameters
    ----------
    reduction : str
        ``"mean"`` or ``"sum"``, as in torch, and it changes the gradient. Default: ``"mean"``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __str__(self) -> str:
        return ErrorFunctionNames.MSE


class CrossEntropy(ErrorFunction):
    """
    Cross Entropy loss function.
    Suitable for classification tasks.
    """

    def __init__(self, with_logits: bool = True, reduction: str = "mean"):
        """
        Parameters
        ----------
        with_logits : bool, optional
            Whether the input to the loss function are logits (unprocessed by activation). Default: True.
        reduction : str
            ``"mean"`` or ``"sum"``, as in torch, and it changes the gradient. Default: ``"mean"``.
        """
        super().__init__(reduction)
        self.with_logits = with_logits

    def __str__(self) -> str:
        return ErrorFunctionNames.SOFTENTROPY if self.with_logits else ErrorFunctionNames.CROSSENTROPY


class SoftEntropy(ErrorFunction):
    """
    Soft Entropy loss function.
    Similar to Cross Entropy but usually applied with a soft layer at the end.

    Parameters
    ----------
    reduction : str
        ``"mean"`` or ``"sum"``, as in torch, and it changes the gradient. Default: ``"mean"``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __str__(self) -> str:
        return ErrorFunctionNames.SOFTENTROPY


__all__ = ["MSE", "CrossEntropy", "SoftEntropy", "ErrorFunction", "ErrorFunctionNames"]
