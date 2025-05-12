"""
    original implementation _inspired by_ Whitney Chiu, who in turn implemented this from
    "Learning Smooth Neural Functions via Lipschitz Regularization" by Liu et al. 2022
    original work found here: https://github.com/whitneychiu/lipmlp_pytorch/blob/main/models/lipmlp.py
        - used under Apache License 2.0
    - extension using the geometric mean described in "Beyond Clear Paths", Section 4.2: https://www.proquest.com/docview/3155972317
    - additional extensions to the LipschitzLinear formulation by me (unpublished) - experiment to follow
"""

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

    # @abstractmethod
    # def initialize_c(self):
    #     raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_lipschitz_constant(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def forward(self, X: torch.Tensor):
        raise NotImplementedError("Subclasses must implement this method.")




class StableSoftplusLipschitzLinear(LipschitzLinearBase):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.c = Parameter(torch.empty(1))
        self.initialize_c()

    def initialize_c(self):
        with torch.no_grad():
            init_val = torch.logsumexp(self.weight.abs(), dim=1).max()
            self.c.data.fill_(init_val)

    def stable_softplus(self, X: torch.Tensor):
        # more numerically stable softplus implementation
        # TODO: consider letting the beta parameter be learnable
        return torch.nn.functional.softplus(X, beta=1, threshold=20)

    def get_lipschitz_constant(self):
        return self.stable_softplus(self.c)

    def forward(self, X: torch.Tensor):
        # normalize lipschitz constant with the infinity norm of the weight matrix (essentially the same as the publication by Liu et al.)
        scale = self.get_lipschitz_constant() / (self.weight.abs().sum(dim=1) + 1e-8)
        scale = scale.clamp(max=1.0)
        return F.linear(X, self.weight * scale.unsqueeze(1), self.bias)



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

    def forward(self, X: torch.Tensor):
        # compute Lipschitz constant lipc
        lipc = self.get_lipschitz_constant()
        #! FIXME: may need to use the infinity or spectral norm of the weight matrix instead of the L1 norm
        scale = lipc / self.weight.abs().sum(dim=1) # normalize with l1 norm of weights
        # TODO: double check the shape of scale here
        scale = scale.clamp(max=1.0)
        return F.linear(X, self.weight * scale.unsqueeze(1), self.bias)



class SpectralNormalizedLinear(LipschitzLinearBase):
    def __init__(self, in_features: int, out_features: int, n_power_iter: int = 10):
        super().__init__(in_features, out_features)
        self: LipschitzLinearBase = torch.nn.utils.parametrizations.spectral_norm(self, "weight", n_power_iterations=n_power_iter)

    def get_lipschitz_constant(self):
        # Lipschitz constant is directly controlled via normalization, not minimized through gradient updates
        #? NOTE: largest singular value of the weight matrix is the Lipschitz constant in this case
        #? NOTE: L2 matrix norm == spectral norm
        return torch.linalg.norm(self.weight, ord=2).detach() # returns the spectral norm of the weight matrix as the Lipschitz constant

    def forward(self, X: torch.Tensor):
        # UPDATED: moved the reinitialization of the module to the constructor since this just registers a constraint applied each time the weight is used
        return F.linear(X, self.weight, self.bias)



class OrthogonalLipschitzLinear(LipschitzLinearBase):
    def __init__(self, in_features: int, out_features: int):
        # initialize through torch.nn.Module since the weight matrix will be parameterized differently
        Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.n_reflectors = min(in_features, out_features)
        # memory optimization: Only allocate vectors of the needed size
        self.is_expansive = self.in_features <= self.out_features
        # for expansive layers (out_features > in_features), use in_features vectors, else use out_features vectors
        col_dim = self.in_features if self.is_expansive else self.out_features
        # initialize Householder vectors (parameterize these instead of weights)
        self.householder_vectors = Parameter(torch.empty(self.n_reflectors, col_dim), requires_grad=True)
        # initialize bias normally
        self.bias = Parameter(torch.empty(out_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        """ initialize Householder vectors with a normal distribution and bias with uniform distribution """
        torch.nn.init.normal_(self.householder_vectors, 0, 0.01)
        # initialize bias the same way as the parent class
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)


    def get_weight_matrix(self):
        """ Compute the effective weight matrix from Householder vectors """
        TOL = 1e-8
        device = self.householder_vectors.device
        if self.is_expansive: # pad identity with zeros if needed (when out_features > in_features)
            weight = torch.eye(self.in_features, device=device)
            padding_dims = (self.out_features - self.in_features, self.in_features)
            weight = torch.cat((weight, torch.zeros(padding_dims, device=device)), dim=0)
        else: # truncate identity if needed (when in_features > out_features)
            weight = torch.eye(self.out_features, self.in_features, device=device)
        # pre-process all vectors at once by taking only the first in_features columns then normalizing them
            #? for contracting layers, this will be all the Householder vectors
        vectors = self.householder_vectors[:, :self.in_features]
        # normalize v for numerical stability with Euclidean norm and small tolerance
        norms = torch.linalg.norm(vectors, dim=1, ord=2, keepdim=True)
        vectors = vectors / (norms + TOL)
        # apply each Householder reflection in sequence
        for i in range(self.n_reflectors):
            v = vectors[i]
            # $$H = I - 2 * vv^T$$
            # weight <- weight @ H      <==>    weight <- weight - 2 * weight @ (vv^T)
            ##projection = torch.matmul(weight, v[:self.in_features].unsqueeze(1)).squeeze(-1)
            ##weight = weight - 2 * torch.outer(projection, v[:self.in_features])
            weight = weight - 2 * torch.outer(weight @ v, v)
        return weight

    def forward(self, X: torch.Tensor):
        """ memory-efficient forward pass using weight matrices computed with Householder reflections """
        weight = self.get_weight_matrix()
        #? retaining the use of F.linear to ensure predictable backend behavior
        return F.linear(X, weight, self.bias)

    def get_lipschitz_constant(self):
        #? NOTE: orthonormal matrices always have spectral norm of 1 and Householder reflections preserve norms
        return torch.tensor(1.0)





#~ Tabling this for now since there's no actual need for it in the linear layers
    #~ for non-linear functions, the Jacobian varies with the input, so the JVP approach will be good for estimating
    #~ local Lipschitz behavior with them later
# class JacobianNormLipschitzLinear(LipschitzLinearBase):
#     def __init__(self, in_features, out_features, num_samples=10):
#         super().__init__(in_features, out_features)
#         self.last_jacobian_norm = None
#         self.num_samples = num_samples

#     def linear_transform(self, X):
#         return F.linear(X, self.weight, self.bias)

#     # TODO: wrap this function in an autograd decorator to avoid calling the function twice when computing the Jacobian
#         # also pretty sure that this will cause autograd problems for the same reason (unless it's implicitly cached by PyTorch)
#         # may be able to do something with torch.compile here as well since it inherits from torch.nn.Module
#     def forward(self, X):
#         # compute and cache Jacobian norm during forward pass without recursion
#         self.last_jacobian_norm = self.empirical_jacobian_norm(X)
#         return self.linear_transform(X)

#     def empirical_jacobian_norm(self, input_sample: torch.Tensor):
#         # TODO: need to write up the empirical Jacobian norm calculation and its connection to the Lipschitz constant
#         input_sample = input_sample.detach().requires_grad_(True)
#         # Store original shape for reshaping
#         original_shape = input_sample.shape
#         feature_dim = original_shape[-1]
#         norm_estimate = 0.0
#         for _ in range(self.num_samples):
#             # create set of random vectors with same shape as input
#             v = torch.randn_like(input_sample)
#             # Reshape to [total_elements, feature_dim] for proper normalization
#             v = v.reshape(-1, feature_dim)
#             # Normalize each vector separately
#             v_norms = torch.norm(v, dim=1, keepdim=True)
#             v /= (v_norms + 1e-8)  # add epsilon for numerical stability
#             # reshape back to original shape
#             v = v.reshape(original_shape)
#             #v /= v.norm() # normalize the random vector
#             jvp = torch.autograd.functional.jvp(self.linear_transform, (input_sample,), (v,), create_graph=False)[1]
#             # compute norm of each output vector
#             jvp_flat = jvp.reshape(-1, jvp.shape[-1])
#             vector_norms = torch.norm(jvp_flat, dim=1)
#             # max norm across all vectors is our estimate - really just the largest singular value of the Jacobian
#             current_estimate = vector_norms.max().item()
#             norm_estimate = max(norm_estimate, current_estimate)
#             # # estimate the operator norm as ||Jv||/||v||
#             # jvp_norm = torch.linalg.norm(jvp, ord=2)
#             # norm_estimate = max(norm_estimate, jvp_norm.item())
#         return norm_estimate

#     def get_lipschitz_constant(self):
#         assert self.last_jacobian_norm is not None, "Jacobian norm has not been computed yet. Run forward first."
#         return self.last_jacobian_norm