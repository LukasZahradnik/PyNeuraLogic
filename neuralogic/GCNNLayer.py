from enum import Enum


class GCNNLayerMode(Enum):
    DEFAULT = 0
    NEURALOGIC = 1
    PYTORCH = 2


class GCNNLayer:
    def __init__(self, mode: GCNNLayerMode = GCNNLayerMode.DEFAULT):
        self.mode = mode

    def forward(self):
        if self.mode in [GCNNLayerMode.DEFAULT, GCNNLayerMode.NEURALOGIC]:
            return
        raise NotImplementedError
