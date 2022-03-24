from neuralogic.core.constructs.metadata import Metadata
from neuralogic.core.enums import Activation, Aggregation
from neuralogic.core.constructs.factories import R, V
from neuralogic.utils.module.module import Module


class APPNPConv(Module):
    def __init__(
        self,
        output_name: str,
        feature_name: str,
        edge_name: str,
        k: int,
        alpha: float,
        activation: Activation = Activation.IDENTITY,
        aggregation: Aggregation = Aggregation.SUM,
    ):
        self.output_name = output_name
        self.feature_name = feature_name
        self.edge_name = edge_name

        self.alpha = alpha
        self.k = k

        self.aggregation = aggregation
        self.activation = activation

    def __call__(self):
        head = R.get(self.output_name)(V.I)
        metadata = Metadata(activation=Activation.IDENTITY, aggregation=self.aggregation)
        edge = R.get(self.edge_name)
        feature = R.get(self.feature_name)

        rules = []
        for k in range(1, self.k):
            k_head = R.get(f"{self.output_name}__{k}")(V.I)
            rules.append((k_head <= feature(V.I)[self.alpha].fixed()) | metadata)

            if k == 1:
                rules.append((k_head <= (feature(V.J)[1 - self.alpha].fixed(), edge(V.J, V.I))) | metadata)
            else:
                rules.append(
                    (k_head <= (R.get(f"{self.output_name}__{k - 1}")(V.J)[1 - self.alpha].fixed(), edge(V.J, V.I)))
                    | metadata
                )
            rules.append(R.get(f"{self.output_name}__{k}") / 1 | Metadata(activation=Activation.IDENTITY))

        if self.k == 1:
            output_rule = head <= (feature(V.J)[1 - self.alpha].fixed(), edge(V.J, V.I))
        else:
            output_rule = head <= (
                R.get(f"{self.output_name}__{self.k - 1}")(V.J)[1 - self.alpha].fixed(),
                edge(V.J, V.I),
            )

        return [
            *rules,
            (head <= feature(V.I)[self.alpha].fixed()) | metadata,
            output_rule | metadata,
            R.get(self.output_name) / 1 | Metadata(activation=self.activation),
        ]
