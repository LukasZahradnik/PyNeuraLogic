from neuralogic import get_neuralogic
from py4j.java_gateway import get_field


class Rule:
    def __init__(self, other: "Rule" = None):
        self.neuralogic = get_neuralogic()
        self.namespace = (
            self.neuralogic.cz.cvut.fel.ida.logic.constructs.template.components
        )

        if other is None:
            self.rule = self.namespace.WeightedRule()
        else:
            self.rule = self.namespace.WeightedRule(other.rule)

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
