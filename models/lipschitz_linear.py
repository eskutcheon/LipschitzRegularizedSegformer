"""
    original implementation by Whitney Chiu, who in turn implemented this from
    "Learning Smooth Neural Functions via Lipschitz Regularization" by Liu et al. 2022
    original work found here: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py
        - used under Apache License 2.0
    - extension using the geometric mean described in "Beyond Clear Paths", Section 4.2: https://www.proquest.com/docview/3155972317
"""

import torch
import math


raise RuntimeError("ERROR: This file is deprecated (kept for reference) and isn't supposed to be called! Terminating...")

class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty(1, requires_grad=True))  # Lipschitz constant
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        # He initialization for better convergence
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        # compute lipschitz constant of initial weight to initialize self.c
        self.c.data = self.weight.data.abs().sum(1).max() # rough initialization with the infinity norm of W
        # might need to add an unsqueeze(0) to self.c.data:
            # self.c.data = W_abs_row_sum.max().unsqueeze(0)

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        # Handle multi-dimensional inputs (batch size, seq_len, features)
        # compute Lipschitz constant lipc
        lipc = self.get_lipschitz_constant()
        scale = lipc / torch.abs(self.weight).sum(1) # normalize with l1 norm of weights
        scale = torch.clamp(scale, max=1.0) # prevent scaling weights by a factor larger than 1
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)


# NOTE: [UNUSED] original implementation of the LipschitzMLP class from the original author's code
class LipschitzMLP(torch.nn.Module):
    def __init__(self, dims):
        """
            dim[0]: input dim
            dim[1:-1]: hidden dims
            dim[-1]: out dim
            assume len(dims) >= 3
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii+1]))
        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        self.relu = torch.nn.ReLU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc *= self.layers[ii].get_lipschitz_constant()
        loss_lipc *= self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x)