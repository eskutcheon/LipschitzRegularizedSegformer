
from models.lipschitz_linear2 import *


LIPSCHITZ_LAYER_REGISTRY = {
    "geometric_mean": GeometricMeanLipschitzLinear,
    "spectral_norm": SpectralNormalizedLinear,
    "stable_softplus": StableSoftplusLipschitzLinear,
    "implicit_layer": ImplicitLipschitzLinear,
    "jacobian_norm": JacobianNormLipschitzLinear,
}


def create_lipschitz_layer(strategy, in_features, out_features):
    layer_cls = LIPSCHITZ_LAYER_REGISTRY.get(strategy)
    if layer_cls is None:
        raise ValueError(f"Lipschitz strategy '{strategy}' is not registered.")
    return layer_cls(in_features, out_features)
