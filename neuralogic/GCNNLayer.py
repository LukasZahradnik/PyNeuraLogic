from enum import Enum
import torch


class GCNNLayerMode(Enum):
    DEFAULT = 0
    NEURALOGIC = 1
    PYTORCH = 2


class GCNNLayer(torch.nn.Module):
    def __init__(self, mode: GCNNLayerMode = GCNNLayerMode.DEFAULT):
        super(GCNNLayer, self).__init__()

        self.mode = mode

    def forward(self):
        if self.mode in [GCNNLayerMode.DEFAULT, GCNNLayerMode.NEURALOGIC]:
            return
        raise NotImplementedError
