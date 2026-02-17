from torch import nn


class DisplacementField(nn.Module):
    def __init__(
        self,
        dim,
        hidden_list,
        activation_fn=None,
    ):
        super().__init__()
        if activation_fn is None:
            activation_fn = nn.ReLU()

        layer_dim_list = [dim] + hidden_list + [dim]
        layers = []
        for idx in range(len(layer_dim_list) - 1):
            layers.append(nn.Linear(layer_dim_list[idx], layer_dim_list[idx + 1]))
            if idx < len(layer_dim_list) - 2:
                layers.append(activation_fn)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
