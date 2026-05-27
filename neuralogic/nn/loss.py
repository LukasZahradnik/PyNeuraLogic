class ErrorFunctionNames:
    MSE = "SQUARED_DIFF"
    CROSSENTROPY = "CROSSENTROPY"
    SOFTENTROPY = "SOFTENTROPY"


class ErrorFunction:
    """
    Base class for error (loss) functions in the neural network.
    """

    pass


class MSE(ErrorFunction):
    """
    Mean Squared Error (SQUARED_DIFF) loss function.
    Suitable for regression tasks.
    """

    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return ErrorFunctionNames.MSE


class CrossEntropy(ErrorFunction):
    """
    Cross Entropy loss function.
    Suitable for classification tasks.
    """

    def __init__(self, with_logits: bool = True):
        """
        Parameters
        ----------
        with_logits : bool, optional
            Whether the input to the loss function are logits (unprocessed by activation). Default: True.
        """
        super().__init__()
        self.with_logits = with_logits

    def __str__(self) -> str:
        return ErrorFunctionNames.SOFTENTROPY if self.with_logits else ErrorFunctionNames.CROSSENTROPY


class SoftEntropy(ErrorFunction):
    """
    Soft Entropy loss function.
    Similar to Cross Entropy but usually applied with a soft layer at the end.
    """

    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return ErrorFunctionNames.SOFTENTROPY


__all__ = ["MSE", "CrossEntropy", "SoftEntropy", "ErrorFunction", "ErrorFunctionNames"]
