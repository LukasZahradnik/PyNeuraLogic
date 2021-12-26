import numpy as np
import jpype


@jpype.JImplements(jpype.JClass("cz.cvut.fel.ida.algebra.functions.ErrorFcn"))
class ErrorFunc:
    def __init__(self, evaluate, differentiate):
        self._evaluate = evaluate
        self._differentiate = differentiate

        self.scalar = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")

    @jpype.JOverride
    def evaluate(self, output, target):
        output = np.array(output.value)
        target = np.array(target.value)

        res = self._evaluate(output, target)
        return self.scalar(res)

    @jpype.JOverride
    def differentiate(self, output, target):
        output = np.array(output.value)
        target = np.array(target.value)

        res = self._differentiate(output, target)
        return self.scalar(res)

    @jpype.JOverride
    def getSingleton(self):
        return None
