class LRDecay:
    def __init__(self):
        self._optimizer = None
        self._decay = None

    def _initialize(self, learning_rate):
        raise NotImplementedError

    def decay(self, epoch: int):
        """
        Manually run the learning rate decay - this is useful when passing sample by sample into the training method
        instead of passing the whole batch of samples. In that case, the decay is not triggered automatically, as it
        is unknown what the current epoch is.

        Parameters
        ----------

        epoch : int
            The number of the current epoch.
        """
        if self._decay is None:
            if self._optimizer is None:
                raise ValueError("LRDecay cannot be initialized - no optimizer has been attached")
            self._optimizer.initialize()
            self._initialize(self._optimizer._lr_object)
        self._decay.decay(epoch)
