class ErrorFunction:
    pass


class MSE(ErrorFunction):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SQUARED_DIFF"


class CrossEntropy(ErrorFunction):
    def __init__(self, with_logits: bool = True):
        super().__init__()
        self.with_logits = with_logits

    def __str__(self):
        return "SOFTENTROPY" if self.with_logits else "CROSSENTROPY"


class SoftEntropy(ErrorFunction):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SOFTENTROPY"
