import math
from typing import Optional, Callable, Any
import torch
from transformers import SegformerForSemanticSegmentation, SegformerForImageClassification, SegformerConfig
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
# from transformers.modeling_outputs import SemanticSegmenterOutput
# local imports
#from models.lipschitz_linear import LipschitzLinear
from .layer_registry import create_lipschitz_layer



class L2SelfAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        # initialize the linear layers for query, key, value, and output projection
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, N, _ = x.size()
        # reshape the input tensor to separate the heads
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)
        # get the pairwise distances between the query and key vectors
        dist = torch.cdist(q, k, p=2)
        # normalize the distances to get the attention weights
        attn = torch.softmax(-dist, dim=-1)
        # perform attention operation (weighted sum of value vectors) then reshape the attention weights to match input
        out = torch.einsum('bhij,bhjd->bhid', attn, v).reshape(B, N, self.dim)
        # apply the output linear projection
        return self.out_proj(out)





#! FIXME: rename this at least - it should be replaced with direct use of the `LipschitzMLP` class but this is the one that actually got
    #! updated to allow for the geometric mean of the Lipschitz constants - LipschitzMLP is almost entirely from the original author's code
class LipschitzSegformerLinear(torch.nn.Module):
    """ Linear Embedding with Lipschitz-regularized linear layer """
    def __init__(self, input_dim: int, output_dim: int, lipschitz_strategy: str):
        super().__init__()
        #self.proj = LipschitzLinear(input_dim, output_dim)
        self.proj = create_lipschitz_layer(lipschitz_strategy, input_dim, output_dim)

    # TODO: need to transition to some builder pattern for setting the particular Lipschitz layer to use

    def get_lipschitz_constant(self):
        # Return the Lipschitz constant of the projection layer
        return self.proj.get_lipschitz_constant()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, N, C)
        return self.proj(hidden_states)  # apply the Lipschitz linear layer and return the output



# this MLP structure is just to keep the original logic of `transformers.models.segformer.modeling_segformer.SegformerDecodeHead` intact
class SegformerLipschitzDecoderHead(SegformerDecodeHead):
    def __init__(self, config: SegformerConfig, lipschitz_strategy: Optional[str] = "geometric_mean"):
        super().__init__(config)
        # TODO: make the default strategy the old infinity norm version from the source paper
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlp_layers = [
            LipschitzSegformerLinear(config.hidden_sizes[i], config.decoder_hidden_size, lipschitz_strategy)
            for i in range(config.num_encoder_blocks)
        ]
        # essentially the only change to the original SegformerDecodeHead is overwriting self.linear_c below
        self.linear_c = torch.nn.ModuleList(mlp_layers)

    def get_lipschitz_loss(self) -> torch.Tensor:
        #loss = torch.prod(torch.tensor([mlp.get_lipschitz_loss() for mlp in self.linear_c]))
        lipschitz_constants = torch.tensor([layer.get_lipschitz_constant() for layer in self.linear_c])
        # return numerically stable geometric mean of the Lipschitz constants
        # TODO: consider setting regularization loss functions in the layers via a higher-order function so regularization follows different strategies
        loss = torch.exp(torch.mean(torch.log(lipschitz_constants)))
        return loss




#!! FIXME: need to iron these classes out using only the new LipschitzLinear layers where applicable

class LRSegformerForSegmentation(SegformerForSemanticSegmentation):
    def replace_decoder_layers(self, lipschitz_strategy: str):
        """ replace the superclass's decoder layers with Lipschitz layers - not the most efficient way to do this but it works """
        del self.decode_head # remove the original decoder head
        self.decode_head = SegformerLipschitzDecoderHead(self.config, lipschitz_strategy)

    def get_lipschitz_loss(self):
        return self.decode_head.get_lipschitz_loss()

    # TODO: add new forward method that calls the superclass's forward method but accepts keyword arguments compatible with the trainer


# TODO: eventually override the base class' constructors (still subclassing SegformerPreTrainedModel) to more efficiently do this - working proof of concept comes first
class LRSegformerForClassification(SegformerForImageClassification):
    def replace_classifier_layer(self, lipschitz_strategy: str):
        """ replace the superclass's classifier layer (a single `torch.nn.Linear`) with a Lipschitz layer """
        # self.classifier is a `torch.nn.Linear` layer in the superclass instantiated with `nn.Linear(config.hidden_sizes[-1], config.num_labels)``
        in_features, out_features = self.classifier.in_features, self.classifier.out_features
        self.classifier = create_lipschitz_layer(lipschitz_strategy, in_features, out_features)

    def get_lipschitz_loss(self):
        return self.classifier.get_lipschitz_constant()
