import math
from abc import ABC, abstractmethod
import torch
from torch.nn import Module, Parameter, functional as F
#from torch.autograd.functional import implicit_jacobian



# TODO: consider letting this inherit from torch.nn.Linear or torch.nn.LazyLinear later
class LipschitzLinearBase(Module, ABC):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = Parameter(torch.empty(out_features, requires_grad=True))
        #~ testing the current code first but think I should add this function in the base class and override it in the subclasses
            #~ the only reason I hesitate is because I think one method will need a whole matrix for each layer, so generalization will be needed
        # self.initialize_c()  # Initialize learnable Lipschitz constant parameter for this layer
        self.reset_parameters()

    def reset_parameters(self):
        # should be the same as the old general parameterization below but this should actually be initialized right
            # stdv = 1. / math.sqrt(self.weight.size(1))
            # self.weight.data.uniform_(-stdv, stdv)
            #? NOTE: this snippet was actually from the original: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #? NOTE: can't remember what this does - it just always accompanies Kaiming initialization
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # reset the Lipschitz constant parameter if it exists (when this function is called outside the constructor)
        if hasattr(self, 'c'):
            self.initialize_c()

    @abstractmethod
    def initialize_c(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def get_lipschitz_constant(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError("Subclasses should implement this method.")



class GeometricMeanLipschitzLinear(LipschitzLinearBase):
    """ Geometric Mean Lipschitz Linear Layer - same as the novel implementation from 'Beyond Clear Paths' """
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.c = Parameter(torch.empty(1))
        self.softplus = torch.nn.Softplus()
        self.initialize_c()

    def initialize_c(self):
        with torch.no_grad():
            # initialize c to the maximum absolute row sum of the weight matrix by default
            self.c.data.fill_(self.weight.abs().sum(dim=1).max().log().item())

    def get_lipschitz_constant(self):
        # Numerically stable formulation - was just `torch.nn.Softplus()` so check if there's a difference later
        return torch.exp(self.c + torch.log1p(torch.exp(-self.c)))

    def forward(self, input):
        # compute Lipschitz constant lipc
        lipc = self.get_lipschitz_constant()
        #! FIXME: may need to use the infinity or spectral norm of the weight matrix instead of the L1 norm
        scale = lipc / self.weight.abs().sum(dim=1) # normalize with l1 norm of weights
        # TODO: double check the shape of scale here
        scale = scale.clamp(max=1.0)
        return F.linear(input, self.weight * scale.unsqueeze(1), self.bias)



class SpectralNormalizedLinear(LipschitzLinearBase):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer('u', torch.randn(out_features))

    def get_lipschitz_constant(self):
        weight_sn, _ = torch.nn.utils.spectral_norm(self.weight, return_sigma=True)
        return torch.norm(weight_sn, 2)

    def forward(self, input):
        weight_sn = torch.nn.utils.spectral_norm(self.weight)
        return F.linear(input, weight_sn, self.bias)



class StableSoftplusLipschitzLinear(LipschitzLinearBase):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.c = Parameter(torch.empty(1))
        self.initialize_c()

    def initialize_c(self):
        with torch.no_grad():
            init_val = torch.logsumexp(self.weight.abs(), dim=1).max()
            self.c.data.fill_(init_val)

    def stable_softplus(self, x):
        # Numerically stable softplus implementation
        return torch.nn.functional.softplus(x, beta=1, threshold=20)

    def get_lipschitz_constant(self):
        return self.stable_softplus(self.c)

    def forward(self, input):
        scale = self.get_lipschitz_constant() / (self.weight.abs().sum(dim=1) + 1e-8)
        scale = scale.clamp(max=1.0)
        return F.linear(input, self.weight * scale.unsqueeze(1), self.bias)



class ImplicitLipschitzLinear(LipschitzLinearBase):
    def __init__(self, in_features, out_features, tol=1e-4, max_iter=50):
        super().__init__(in_features, out_features)
        self.tol = tol
        self.max_iter = max_iter

    def implicit_function(self, z, x):
        # Implicit formulation: f(z, x) = z - (Wx + b)
        return z - (F.linear(x, self.weight, self.bias))

    def forward(self, input):
        z = input.clone().detach().requires_grad_(True)
        for _ in range(self.max_iter):
            with torch.enable_grad():
                f_z = self.implicit_function(z, input)
                if f_z.norm() < self.tol:
                    break
                # TODO: should maybe consider `torch.autograd.functional.implicit_jacobian`
                    # also learned torch.autograd.functional.hessian exists
                J = torch.autograd.functional.jacobian(lambda z_: self.implicit_function(z_, input), z)
                update = torch.linalg.solve(J, -f_z)
                z = (z + update).detach().requires_grad_(True)
        return z

    def get_lipschitz_constant(self):
        # approximate Lipschitz by spectral norm of weight matrix (order 2 is spectral norm for matrices)
        return torch.linalg.norm(self.weight, ord=2)



class JacobianNormLipschitzLinear(LipschitzLinearBase):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def empirical_jacobian_norm(self, input_sample):
        input_sample.requires_grad_(True)
        output_sample = self.forward(input_sample)
        jacobian = torch.autograd.functional.jacobian(lambda x: self.forward(x), input_sample)
        norm = jacobian.norm(p='fro')
        return norm

    def get_lipschitz_constant(self, input_sample=None):
        if input_sample is None:
            raise ValueError("Jacobian norm requires an input sample.")
        return self.empirical_jacobian_norm(input_sample)
