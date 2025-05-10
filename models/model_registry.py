
# from abc import ABC, abstractmethod
# import torch
# import logging
# TODO: need to update A LOT of other aspects to comply with the old `transformers` codebase for drop-in replacement
from transformers.models.segformer import (
    SegformerForSemanticSegmentation,
    SegformerForImageClassification,
    SegformerConfig
)
# local imports
from .segformer import LRSegformerForClassification, LRSegformerForSegmentation
# from transformers import logger as hf_logger
# hf_logger.setLevel(logging.ERROR)  # Suppress warnings from transformers


#! should only be used for training - loading Lipschitz models for inference will have different model parameters
def get_segformer_model(task, model_type, pretrained_name, num_labels=None, lipschitz_strategy=None):
    assert task in ['segmentation', 'classification'], "Task must be either 'segmentation' or 'classification'"
    if model_type == 'baseline':
        model_cls = SegformerForSemanticSegmentation if task == 'segmentation' else SegformerForImageClassification
        return model_cls.from_pretrained(pretrained_name, num_labels=num_labels)
    elif model_type == 'lipschitz':
        return lipschitz_segformer_factory(task, lipschitz_strategy, pretrained_name, num_labels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'baseline' and 'lipschitz'.")



def lipschitz_segformer_factory(task: str, lipschitz_strategy: str, pretrained_name: str, num_labels: int):
    model = None
    config = SegformerConfig.from_pretrained(pretrained_name, num_labels=num_labels)
    if task == 'segmentation':
        model = LRSegformerForSegmentation.from_pretrained(pretrained_name, config=config)
        model.replace_decoder_layers(lipschitz_strategy)
    elif task == 'classification':
        model = LRSegformerForClassification.from_pretrained(pretrained_name, config=config)
        model.replace_classifier_layer(lipschitz_strategy)
    else:
        raise ValueError(f"Unsupported task: {task}. Supported tasks are 'segmentation' and 'classification'.")
    return model