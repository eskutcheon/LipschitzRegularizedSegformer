
import torch
from transformers.models.segformer import SegformerForSemanticSegmentation, SegformerForImageClassification
# local imports
from .segformer import LRSegformerForClassification, LRSegformerForSegmentation
from .layer_registry import create_lipschitz_layer



SEGFORMER_MODEL_REGISTRY = {
    "baseline": {
        "segmentation": SegformerForSemanticSegmentation,
        "classification": SegformerForImageClassification
    },
    "lipschitz": {
        "segmentation": LRSegformerForSegmentation,
        "classification": LRSegformerForClassification
    }
    # TODO: add variant for using Lipschitz layers in the encoder (attention modules) as well
}


#! should only be used for training - loading Lipschitz models for inference will have different model parameters
def get_segformer_model(task, model_type, pretrained_name, num_classes=None, lipschitz_strategy=None):
    assert task in ['segmentation', 'classification'], "Task must be either 'segmentation' or 'classification'"
    assert model_type in ['baseline', 'lipschitz'], "Model type must be either 'baseline' or 'lipschitz'"
    assert not (model_type == "baseline" and lipschitz_strategy), "Lipschitz strategy should not be specified for baseline models"
    if model_type == 'baseline':
        model_cls = SEGFORMER_MODEL_REGISTRY['baseline'][task]
        model = model_cls.from_pretrained(pretrained_name, num_labels=num_classes)
    elif model_type == 'lipschitz':
        model_cls = SEGFORMER_MODEL_REGISTRY['lipschitz'][task]
        
        model = model_cls.from_pretrained(pretrained_name, num_labels=num_classes)
        # Replace the linear layers with Lipschitz layers if specified
        if lipschitz_strategy:
            pass
            # TODO: need to specify the layer type to finish initializing the model - may need to move to a builder pattern
            # for name, module in model.named_modules():
            #     if isinstance(module, torch.nn.Linear):
            #         in_features = module.in_features
            #         out_features = module.out_features
            #         lipschitz_layer = create_lipschitz_layer(lipschitz_strategy, in_features, out_features)
            #         setattr(model, name, lipschitz_layer)
    return model