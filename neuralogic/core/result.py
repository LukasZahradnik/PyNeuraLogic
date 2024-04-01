import json


class Result:
    __slots__ = "_result", "_sample", "_model", "_number_format"

    def __init__(self, result, sample, model, number_format):
        self._result = result
        self._sample = sample
        self._model = model
        self._number_format = number_format

    def backward(self):
        self._model._trainer.backpropSample(self._model._backpropagation, self._result, self._sample)

    def value(self):
        return json.loads(str(self._result.getOutput().toString(self._number_format)))


class Results:
    __slots__ = "results"

    def __init__(self, results):
        self.results = results

    def backward(self):
        for result in self.results:
            result.backward()

    def values(self):
        return [result.value() for result in self.results]
