from py4j.java_gateway import get_field

from antlr4 import InputStream, CommonTokenStream
from .grammar import NeuralogicLexer, NeuralogicParser
from .error import InvalidRuleException


class Rule:
    @staticmethod
    def from_str(rule: str) -> "Rule":
        lexer = NeuralogicLexer(InputStream(rule))
        stream = CommonTokenStream(lexer)
        parser = NeuralogicParser(stream)
        parser.examplesFile()

        if parser.getNumberOfSyntaxErrors() != 0:
            raise InvalidRuleException

        return Rule(rule)

    def __init__(self, rule: str):
        self.original_string = rule

    @property
    def weight(self):
        return get_field(self.rule, "weight")

    @weight.setter
    def weight(self, x):
        pass

    @property
    def offset(self):
        return get_field(self.rule, "offset")

    @offset.setter
    def offset(self, x):
        pass

    @property
    def head(self):
        return get_field(self.rule, "head")

    @head.setter
    def head(self, x):
        pass

    @property
    def aggregation_function(self):
        return get_field(self.rule, "aggregationFcn")

    @aggregation_function.setter
    def aggregation_function(self, x):
        pass

    @property
    def activation_function(self):
        return get_field(self.rule, "activationFcn")

    @activation_function.setter
    def activation_function(self, x):
        pass

    @property
    def metadata(self):
        return get_field(self.rule, "metadata")

    @metadata.setter
    def metadata(self, x):
        pass
