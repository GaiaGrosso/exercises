import torch
import torch.nn as nn

# Define Taylor expansion model: f(x, nv) = a_1(x) * nv + a_2(x) * nv^2 + ...
class PolynomialModel(nn.Module):
    def __init__(self, input_dim, polyn_degree=1):
        super(PolynomialModel, self).__init__()
        self.polyn_degree = polyn_degree
        self.networks = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, 13), 
                                                     nn.ReLU(), 
                                                     nn.Linear(13, 1)) 
                                       for _ in range(polyn_degree)])

    def forward(self, x, nv):
        f_components = [self.networks[i](x)[:, 0]* nv[:, 0]**(i+1) for i in range(self.polyn_degree)] # Pass through all layers
        f_components = torch.stack(f_components, dim=0) # [polyn_degree, N, ]
        f = torch.sum(f_components, dim=0) # [N, ]
        return f.reshape((-1, 1)) # [N, 1]

    def get_coeffs(self, x):
        f_components = [self.networks[i](x) for i in range(self.polyn_degree)] # Pass through all layers
        return torch.stack(f_components, dim=0) # [polyn_degree, N, 1]
    
    def loss(self, pred, target):
        y = target[:, 0]
        return torch.sum((1-y)*(torch.exp(pred)-1) - y*pred)


class PolynomialModel(nn.Module):
    """
    Polynomial model:
        f(x, nu) = sum_k  f_k(x) * nu^(k+1)

    Each f_k(x) is represented by an independent neural network
    whose architecture is configurable.
    """

    def __init__(self, input_dim, polyn_degree=1, hidden_layers=(13,), clip=None):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input features x
        polyn_degree : int
            Number of polynomial components
        hidden_layers : tuple or list
            Number of nodes in each hidden layer
            e.g. (13,)       -> 1 hidden layer
                 (32, 16)    -> 2 hidden layers
        """
        super().__init__()
        self.weight_clipping=clip
        self.polyn_degree = polyn_degree
        self.hidden_layers = hidden_layers

        # Build one network per polynomial component
        self.networks = nn.ModuleList(
            [self._build_network(input_dim) for _ in range(polyn_degree)]
        )

    def _build_network(self, input_dim):
        """
        Builds a single MLP:
            input_dim -> hidden_layers -> 1
        """
        layers = []
        in_dim = input_dim

        for h in self.hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # output layer
        return nn.Sequential(*layers)

    def forward(self, x, nv):
        """
        x  : [N, input_dim]
        nv : [N, 1]  (nuisance parameter)

        Returns
        -------
        f : [N, 1]
        """
        f_components = [
            self.networks[i](x)[:, 0] * (nv[:, 0] ** (i + 1))
            for i in range(self.polyn_degree)
        ]

        f_components = torch.stack(f_components, dim=0)  # [polyn_degree, N]
        f = torch.sum(f_components, dim=0)               # [N]

        return f.unsqueeze(1)  # [N, 1]

    def get_coeffs(self, x):
        """
        Returns individual polynomial coefficients f_k(x)

        Output shape: [polyn_degree, N, 1]
        """
        f_components = [self.networks[i](x) for i in range(self.polyn_degree)]
        return torch.stack(f_components, dim=0).squeeze(-1).transpose(0, 1)

    def loss(self, pred, target):
        """
        pred   : [N, 1]
        target : [N, 2]   (y, weight)
        """
        f = pred[:, 0]
        y = target[:, 0]
        return torch.sum((1 - y) * (torch.exp(f) - 1) - y * f)

    def clip_weights(self):
        """
        Hard projection of parameters onto [-c, c].
        Call after optimizer.step().
        """
        if self.weight_clipping is None:
            return

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.clamp_(-self.weight_clipping, self.weight_clipping)
                    if m.bias is not None:
                        m.bias.clamp_(-self.weight_clipping, self.weight_clipping)