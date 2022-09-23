class ErrorFunctionNames:
    MSE = "SQUARED_DIFF"
    CROSSENTROPY = "CROSSENTROPY"
    SOFTENTROPY = "SOFTENTROPY"


class ErrorFunction:
    pass


class MSE(ErrorFunction):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return ErrorFunctionNames.MSE


class CrossEntropy(ErrorFunction):
    def __init__(self, with_logits: bool = True):
        super().__init__()
        self.with_logits = with_logits

    def __str__(self):
        return ErrorFunctionNames.SOFTENTROPY if self.with_logits else ErrorFunctionNames.CROSSENTROPY


class SoftEntropy(ErrorFunction):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return ErrorFunctionNames.SOFTENTROPY


__all__ = ["MSE", "CrossEntropy", "SoftEntropy", "ErrorFunction", "ErrorFunctionNames"]
