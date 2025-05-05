import math
from typing import Optional, Callable, Any
import torch
from transformers import SegformerForSemanticSegmentation, SegformerForImageClassification
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
# from transformers.modeling_outputs import SemanticSegmenterOutput
#from models.lipschitz_linear import LipschitzLinear




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





#! FIXME: rename this at least - it should be replace with direct use of the `LipschitzMLP` class but this is the one that actually got
    #! updated to allow for the geometric mean of the Lipschitz constants - LipschitzMLP is almost entirely from the original author's code
class LipschitzSegformerLinear(torch.nn.Module):
    """ Linear Embedding with Lipschitz-regularized linear layer """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = LipschitzLinear(input_dim, output_dim)

    # TODO: need to transition to some builder pattern for setting the particular Lipschitz layer to use

    def get_lipschitz_loss(self):
        # Return the Lipschitz constant of the projection layer
        return self.proj.get_lipschitz_constant()

    def forward(self, hidden_states):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, N, C)
        hidden_states = self.proj(hidden_states)  # Apply the Lipschitz linear layer
        return hidden_states



# this MLP structure is just to keep the original logic of `transformers.models.segformer.modeling_segformer.SegformerDecodeHead` intact
class SegformerLipschitzDecoderHead(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        #^ UPDATE: replaced the SegformerMLP instances with LipschitzSegformerLinear (exclusively for the decoder layers, not the other feedforward layers)
        mlps = []
        for i in range(config.num_encoder_blocks): # default num_encoder_blocks is 4
            mlp = LipschitzSegformerLinear(
                input_dim=config.hidden_sizes[i],
                output_dim=config.decoder_hidden_size
            )
            mlps.append(mlp)
        # essentially the only change to the original SegformerDecodeHead is overwriting self.linear_c below
        self.linear_c = torch.nn.ModuleList(mlps)

    def get_lipschitz_loss(self):
        #loss = torch.prod(torch.tensor([mlp.get_lipschitz_loss() for mlp in self.linear_c]))
        lipschitz_constants = torch.tensor([mlp.get_lipschitz_loss() for mlp in self.linear_c])
        # return numerically stable geometric mean of the Lipschitz constants
        loss = torch.exp(torch.mean(torch.log(lipschitz_constants), dim=0))
        #print("lipschitz loss (geometric mean of constants) before scaling: ", loss)
        return loss






#!! FIXME: need to iron these classes out using only the new LipschitzLinear layers where applicable

class LRSegformerForSegmentation(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        #^ added the following to overwrite the original decode_head with the Lipschitz version
        #self.decode_head = SegformerLipschitzDecoderHead(config)

    def set_decoder(self, ):
        # needs to reset `self.decode_head` from the original Segformer model to the Lipschitz version
        pass

    def get_lipschitz_loss(self):
        return self.decode_head.get_lipschitz_loss()


class LRSegformerForClassification(SegformerForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        #^ added the following to overwrite the original decode_head with the Lipschitz version
        self.classifier = LipschitzLinear(config.hidden_sizes[-1], config.num_labels)
        #self.decode_head = SegformerLipschitzDecoderHead(config)

    def set_decoder(self, decoder_layer: Optional[Callable] = None):
        # needs to reset `self.classifier` from the original Segformer model to the Lipschitz version
            # self.classifier is just a single linear layer for output logits
        # self.classifier =
        pass

    def get_lipschitz_loss(self):
        return self.classifier.get_lipschitz_loss()
