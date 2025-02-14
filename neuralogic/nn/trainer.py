from neuralogic.core.neural_module import NeuralModule


class Trainer:
    def __init__(self, module: NeuralModule):
        self._module = module

    def train(self, dataset, epochs):
        return self._module.train(dataset, epochs)

    def test(self, dataset):
        return self._module.test(dataset)
