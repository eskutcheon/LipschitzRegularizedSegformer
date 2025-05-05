import torch
from transformers import SegformerImageProcessor
#from models import SegformerLCNSemanticSegmentation, SegformerLCNImageClassification
from ..models.segformer import LRSegformerForSegmentation, LRSegformerForClassification
from .train import train
from .dataset import get_dataloader



# TODO: add small argparser to specify the task (segmentation or classification) and whether to train or infer
def get_model_and_processor(task, pretrained_name, num_classes=None):
    assert task in ['segmentation', 'classification'], "Task must be either 'segmentation' or 'classification'"
    if task == 'segmentation':
        model = LRSegformerForSegmentation.from_pretrained(pretrained_name, num_labels=num_classes)
    elif task == 'classification':
        model = LRSegformerForClassification.from_pretrained(pretrained_name, num_labels=num_classes)
    processor = SegformerImageProcessor.from_pretrained(pretrained_name)
    return model, processor



if __name__ == "__main__":
    PRETRAINED_WEIGHTS = "nvidia/mit-b0" # 'nvidia/segformer-b0-finetuned-ade-512-512'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, processor = get_model_and_processor('segmentation', PRETRAINED_WEIGHTS, num_classes=150)
    model = model.to(device) # TODO: wrap with DataParallel if multiple GPUs are available
    # Data loaders
    train_loader = get_dataloader('scene_parse_150', processor, 'train', task='segmentation')
    val_loader = get_dataloader('scene_parse_150', processor, 'validation', task='segmentation')
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # Metrics to track
    metric_classes = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'phi']
    num_classes = 150  # Adjust according to dataset used
    # Train the model
    train(model, train_loader, val_loader, epochs=10, device=device, optimizer=optimizer,
        metric_classes=metric_classes, num_classes=num_classes)
