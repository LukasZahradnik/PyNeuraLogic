import torch
import torch.nn.functional as F
import torch_geometric

from neuralogic.core.settings import Activation, Aggregation
from neuralogic.utils.templates import TemplateList, GINConv, SAGEConv, GCNConv, Pooling

native_activations = {
    str(Activation.RELU): F.relu,
    str(Activation.TANH): F.tanh,
    str(Activation.SIGMOID): F.sigmoid,
    str(Activation.IDENTITY): lambda x: x,
    str(Activation.SOFTMAX): F.softmax,
    # TODO: Add all activations
}

native_aggregations = {
    str(Aggregation.AVG): torch_geometric.nn.global_mean_pool,
    str(Aggregation.SUM): torch_geometric.nn.global_add_pool,
    str(Aggregation.MAX): torch_geometric.nn.global_max_pool,
}


class NeuraLogic(torch.nn.Module):
    def __init__(self, module_list: TemplateList):
        super().__init__()

        self.modules = []
        self.evaluations = []

        input_shape = module_list.num_features

        if input_shape is None:
            raise Exception

        for i, module in enumerate(module_list.modules):
            activation = str(module.activation)
            if activation not in native_activations:
                raise Exception
            activation_fun = native_activations[activation]
            out_channels = 1 if i + 1 == len(module_list.modules) else module.in_channels

            if isinstance(module, GCNConv):
                self.modules.append(
                    torch_geometric.nn.GCNConv(input_shape, out_channels, normalize=False, cached=False, bias=False)
                )
            elif isinstance(module, SAGEConv):
                self.modules.append(torch_geometric.nn.SAGEConv(input_shape, out_channels, normalize=False, bias=False))
            elif isinstance(module, GINConv):
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(input_shape, module.in_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(module.in_channels, out_channels),
                )

                self.modules.append(torch_geometric.nn.GINConv(mlp))
            elif isinstance(module, Pooling):
                pooling_layers = []

                for _ in module.layers:
                    layer = torch.nn.Linear(input_shape, out_channels, bias=False)
                    pooling_layers.append(layer)
                    self.modules.append(layer)

                def _pooling(x, edge_index, batch, xs):
                    pooling_xs = [
                        self.modules[i + j](native_aggregations[str(module.aggregation)](xs[layer], batch))
                        for j, layer in enumerate(module.layers)
                    ]

                    pooling_x = torch.stack(pooling_xs, dim=0)
                    return native_activations[str(module.activation)](torch.sum(pooling_x, dim=0))

                self.evaluations.append(_pooling)
                input_shape = module.in_channels
                continue
            else:
                raise Exception

            self.evaluations.append(lambda x, edge_index, batch, xs: activation_fun(self.modules[i](x, edge_index)))
            input_shape = module.in_channels

    def forward(self, x, edge_index, batch):
        xs = []

        for module in self.modules:
            x = module(x, edge_index, batch, xs)
            xs.append(x)
        return x
