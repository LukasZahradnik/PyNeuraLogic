import jpype

from neuralogic.optim.lr_scheduler.lr_decay import LRDecay


class ArithmeticLR(LRDecay):
    r"""
    Decay learning rate on every epoch by the following formula

    .. math::

        \mathbf{lr}_i = \mathbf{lr}_{i-1} - \dfrac{\mathbf{lr}_{0}}{max\_steps}

    Parameters
    ----------

    max_steps : int
    """

    def __init__(self, max_steps: int):
        super().__init__()
        self.max_steps = max_steps

    def _initialize(self, learning_rate):
        class_name = "cz.cvut.fel.ida.neural.networks.computation.training.strategies.Hyperparameters.ArithmeticDecay"
        self._decay = jpype.JClass(class_name)(learning_rate, self.max_steps)

        return self._decay

    def __str__(self) -> str:
        return f"ArithmeticLR(max_steps={self.max_steps})"
