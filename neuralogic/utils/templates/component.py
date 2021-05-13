from neuralogic.core import Template


class AbstractComponent:
    features_name = "node_features"

    def __init__(self, *, weight_shape, has_edge_attrs=True):
        self.weight_shape = weight_shape
        self.has_edge_attrs = has_edge_attrs

    def build(self, template: Template, layer_count: int, previous_name: str) -> str:
        pass
