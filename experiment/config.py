

ALL_METRICS = ['iou', 'accuracy', 'precision', 'recall', 'f1', 'phi'] # all metrics to track during training and evaluation
CHECKPOINT_DIR = "checkpoints" # directory to save checkpoints

PRETRAINED_WEIGHTS = "nvidia/mit-b0" # baseline 'nvidia/segformer-b0-finetuned-ade-512-512' pre-trained model weights for all models

SEGMENTATION_TRAIN_DATASET = "zhoubolei/scene_parse_150" # dataset for training segmentation model
# TODO: change later to something with classification labels but that isn't same the initial training was done on
CLASSIFICATION_TRAIN_DATASET = "ILSVRC/imagenet-1k"      # dataset for training classification model