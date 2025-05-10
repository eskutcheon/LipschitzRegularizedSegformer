
from typing import Union
from models.lipschitz_linear import *


LIPSCHITZ_LAYER_REGISTRY = {
    "geometric_mean": GeometricMeanLipschitzLinear,
    "spectral_norm": SpectralNormalizedLinear,
    "stable_softplus": StableSoftplusLipschitzLinear,
    # "jacobian_norm": JacobianNormLipschitzLinear,
    "orthogonal": OrthogonalLipschitzLinear,
}

# TODO: create this type more dynamically with the keys of LIPSCHITZ_LAYER_REGISTRY to avoid hardcoding the types
LipschitzLayerType = Union[
    GeometricMeanLipschitzLinear,
    SpectralNormalizedLinear,
    StableSoftplusLipschitzLinear,
    # JacobianNormLipschitzLinear,
    OrthogonalLipschitzLinear
]

# TODO: add type annotations with a new type for the layer classes (in new file)
def create_lipschitz_layer(strategy: str, in_features: int, out_features: int) -> LipschitzLayerType:
    layer_cls = LIPSCHITZ_LAYER_REGISTRY.get(strategy, None)
    if layer_cls is None:
        raise ValueError(f"Lipschitz strategy '{strategy}' is not registered.")
    return layer_cls(in_features, out_features)
