import jpype

from neuralogic.optim.lr_scheduler.lr_decay import LRDecay


class GeometricLR(LRDecay):
    r"""
    Decay learning rate on every :math:`steps` epoch by the following formula

    .. math::

        \mathbf{lr}_i = \mathbf{lr}_{i-1} \cdot decay\_rate

    Parameters
    ----------

    decay_rate : float

    steps : int

    """

    def __init__(self, decay_rate: float, steps: int):
        super().__init__()
        self.decay_rate = decay_rate
        self.steps = steps

    def _initialize(self, learning_rate):
        class_name = "cz.cvut.fel.ida.neural.networks.computation.training.strategies.Hyperparameters.GeometricDecay"
        self._decay = jpype.JClass(class_name)(learning_rate, self.decay_rate, self.steps)

        return self._decay

    def __str__(self) -> str:
        return f"GeometricLR(decay_rate={self.decay_rate}, steps={self.steps})"
