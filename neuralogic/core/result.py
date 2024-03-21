class Result:
    __slots__ = "result", "sample", "model", "number_format"

    def __init__(self, result, sample, model, number_format):
        self.result = result
        self.sample = sample
        self.model = model
        self.number_format = number_format

    def backward(self):
        weight_updater = self.model.trainer.backpropSample(self.model.backpropagation, self.result, self.sample)
        self.model.trainer.updateWeights(self.model.strategy.getCurrentModel(), weight_updater)

        self.model.trainer.invalidateSample(self.model.invalidation, self.sample)

    def value(self):
        return self.result.getOutput().toString(self.number_format)


class Results:
    __slots__ = "results"

    def __init__(self, results):
        self.results = results

    def backward(self):
        for result in self.results:
            result.backward()

    def values(self):
        return [result.value() for result in self.results]
