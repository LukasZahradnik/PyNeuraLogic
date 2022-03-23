from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.utils.module.module import Module


class TAGConv(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_name: str,
        feature_name: str,
        edge_name: str,
        k: int = 2,
        activation: Activation = Activation.IDENTITY,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.k = k
        self.activation = activation

    def __call__(self):
        metadata = Metadata(activation=Activation.IDENTITY, aggregation=Aggregation.SUM)
        head = R.get(self.output_name)
        feature = R.get(self.feature_name)
        edge = R.get(self.edge_name)

        hop_rules = []

        for i in range(self.k + 1):
            hop_rules.append(
                (
                    head(V.X0)
                    <= (
                        feature(f"X{i}")[self.out_channels, self.in_channels],
                        *(edge(f"X{b}", f"X{a}") for a, b in zip(range(i + 1), range(1, i + 2))),
                    )
                )
                | metadata
            )

        return [
            *hop_rules,
            head / 1 | Metadata(activation=self.activation),
        ]
