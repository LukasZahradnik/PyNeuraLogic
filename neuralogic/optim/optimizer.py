class Optimizer:
    @property
    def lr(self) -> float:
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__

    def is_default(self) -> bool:
        return True
