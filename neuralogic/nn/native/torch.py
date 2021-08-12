import torch
import torch_geometric

from neuralogic.core.settings import Activation, Aggregation
from neuralogic.utils.templates import TemplateList, GINConv, SAGEConv, GCNConv, GlobalPooling, Embedding

native_activations = {
    str(Activation.RELU): torch.relu,
    str(Activation.TANH): torch.tanh,
    str(Activation.SIGMOID): torch.sigmoid,
    str(Activation.IDENTITY): lambda x: x,
    str(Activation.SOFTMAX): torch.softmax,
}

native_aggregations = {
    str(Aggregation.AVG): torch_geometric.nn.global_mean_pool,
    str(Aggregation.SUM): torch_geometric.nn.global_add_pool,
    str(Aggregation.MAX): torch_geometric.nn.global_max_pool,
}


class NeuraLogic(torch.nn.Module):
    def __init__(self, module_list: TemplateList):
        super().__init__()

        self.module_list = torch.nn.ModuleList()
        self.evaluations = []

        for i, module in enumerate(module_list.modules):
            activation = str(module.activation)
            if activation not in native_activations:
                raise Exception
            activation_fun = native_activations[activation]

            if isinstance(module, Embedding):
                self.module_list.append(torch.nn.Embedding(module.num_embeddings, module.embedding_dim))
                self.evaluations.append(lambda x, edge_index, batch, xs, i=i: self.module_list[i](x))

                continue
            elif isinstance(module, GCNConv):
                self.module_list.append(
                    torch_geometric.nn.GCNConv(
                        module.in_channels, module.out_channels, normalize=False, cached=False, bias=False
                    )
                )
            elif isinstance(module, SAGEConv):
                self.module_list.append(
                    torch_geometric.nn.SAGEConv(module.in_channels, module.out_channels, normalize=False, bias=False)
                )
            elif isinstance(module, GINConv):
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(module.in_channels, module.out_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(module.out_channels, module.out_channels),
                )

                self.module_list.append(torch_geometric.nn.GINConv(mlp))
            elif isinstance(module, GlobalPooling):
                pooling_layers = []

                for _ in module.jumping_knowledge:
                    layer = torch.nn.Linear(module.in_channels, module.out_channels, bias=False)
                    pooling_layers.append(layer)
                    self.module_list.append(layer)

                def _pooling(x, edge_index, batch, xs, i=i, module=module):
                    pooling_xs = [
                        self.module_list[i + j](native_aggregations[str(module.aggregation)](xs[layer], batch))
                        for j, layer in enumerate(module.jumping_knowledge)
                    ]

                    pooling_x = torch.stack(pooling_xs, dim=0)
                    return native_activations[str(module.activation)](torch.sum(pooling_x, dim=0))

                self.evaluations.append(_pooling)
                continue
            else:
                raise Exception

            def evaluate(x, edge_index, batch, xs, i=i, activation_fun=activation_fun):
                return activation_fun(self.module_list[i](x, edge_index))

            self.evaluations.append(evaluate)

    def forward(self, x, edge_index, batch=None):
        xs = []

        for evaluation in self.evaluations:
            x = evaluation(x, edge_index, batch, xs)
            xs.append(x)
        return x
