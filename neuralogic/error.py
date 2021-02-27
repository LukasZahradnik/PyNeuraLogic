class InvalidRuleException(Exception):
    pass


class DatasetAlreadyLoadedException(Exception):
    pass


class MixedActivationFunctionsInLayerException(Exception):
    pass


class MixedWeightsAndNoWeightsInLayerException(Exception):
    pass
