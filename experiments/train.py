

from typing import Dict, List, Tuple, Optional, Any
from warnings import catch_warnings, simplefilter
from dataclasses import dataclass, field
from PIL.Image import Image
import torch
import torchvision.transforms.v2 as TT
from transformers import Trainer, TrainingArguments, TrainerCallback, EvalPrediction


# TODO: look into trying later to reduce memory:
# from transformers import AdamW8bit
# optimizer = AdamW8bit(model.parameters(), lr=args.lr)
# trainer.create_optimizer = lambda: optimizer



#  custom TrainingArguments object with an extra field for the Lipschitz coefficient
@dataclass
class CustomTrainingArguments(TrainingArguments):
    lambda_lip: float = field(
        default=0.1,
        metadata={"help": "Weight for the Lipschitz regulariser."},
    )

#? NOTE: max_steps is required when using a streaming dataset, but it's derived heuristically without the dataset size here
def get_training_args(output_dir, max_steps, num_epochs=30, learning_rate=5e-5, batch_size=8, lambda_lip=0.1, **kwargs):
    return CustomTrainingArguments(
        do_train=True,
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        dataloader_pin_memory=False, # set to False since the tensors are moved to GPU in the collate function
        #gradient_accumulation_steps=batch_size//2,
        #dataloader_prefetch_factor=2,
        #gradient_checkpointing = True,
        #remove_unused_columns=False,   # may be needed later for datasets with different column names than what the model `forward` method expects
        #dataloader_num_workers=4,      # have to keep this at 0 for streaming datasets
        learning_rate=learning_rate,
        # NOTE: can also set adam_beta1, adam_beta2, adam_epsilon, weight_decay, warmup_steps, etc.
        fp16=True, # use 16-bit mixed-precision float during training
        logging_strategy="epoch",
        #batch_eval_metrics=True,       # may not want to do this
        # logging_steps=10,
        # save_steps=1000,
        # evaluation_strategy="steps",
        # eval_steps=1000,
        save_total_limit=1,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # report_to="tensorboard",
        lambda_lip=lambda_lip,          # used within HuggingFaceModelTrainer.compute_loss() as self.args.lambda_lip
        **kwargs
    )


# # TODO: set in the trainer with `add_callback` or pass it to the TrainingArguments instance
class MetricTrackerCallback(TrainerCallback):
    def __init__(self, metric_names, num_classes, device):
        from experiments.metrics import get_MetricTracker
        self.tracker = get_MetricTracker(metric_names, num_classes, device)

    def on_evaluate(self, args, state, control, **kwargs):
        preds, labels = kwargs["metrics"]["eval_predictions"]  # returned by compute_metrics
        self.tracker.reset()
        self.tracker.update(torch.tensor(preds), torch.tensor(labels))
        metrics = self.tracker.compute()
        print(f"MetricTracker: {metrics}")


def build_compute_metrics_fn(num_labels: int, task: str):
    """ Returns a function that Trainer will call every eval step """
    from .metrics import get_MetricTracker
    import experiments.config as cfg
    tracker = get_MetricTracker(cfg.ALL_METRICS, num_labels, device="cpu")
    #def _fn(eval_pred: EvalPrediction, compute_result: bool) -> Dict[str, float]: # if using
    def _fn(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        with torch.no_grad():
            #! FIXME: I'm like 99% sure that this isn't right - I think the EvalPrediction preserves the tensors' types and they should be pytorch tensors
                #! also I think the logits still need a softmax application before argmax
                #! also check the dimensions to be sure that the logits are one-hot encoded with a channel dimension
            if task == "segmentation":
                preds = torch.from_numpy(logits).argmax(1)
                labels = torch.from_numpy(labels)
            else:  # classification
                preds = torch.from_numpy(logits).argmax(1)
                labels = torch.from_numpy(labels)
            tracker.reset()
            tracker.update(preds, labels)
            return {k: v.item() for k, v in tracker.compute().items()}
    return _fn



class CustomImageProcessor:
    """ wrote this just because I hate that SegformerImageProcessor so damn much
        - technically not as robust to various inputs but it's a lot more transparent and tailored to these experiments
    """
    # default mean and standard deviation (along RGB channels) for ImageNet for normalization
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]
    #DEFAULT_BACKGROUND_LABEL = 0
    def __init__(self,
                 do_resize: bool = True,
                 do_rescale: bool = True,
                 do_normalize: bool = True,
                 do_reduce_labels: bool = False,
                 size: Tuple[int, int] = (256, 256),
                 data_mean: List[float] = None,
                 data_std: List[float] = None,
                 #background_label: int = None,
    ):
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_reduce_labels = do_reduce_labels
        self.size = size
        self.data_mean = data_mean or self.DEFAULT_MEAN
        self.data_std = data_std or self.DEFAULT_STD
        #self.background_label = background_label or self.DEFAULT_BACKGROUND_LABEL


    def _to_tensor(self, image: Image, is_mask: bool = False) -> torch.Tensor:
        # resize and convert PIL image to tensor
        # resize before conversion to tensor since the PIL backend is typically faster
        if self.do_resize:
            resize_kw = {
                "interpolation": TT.InterpolationMode.NEAREST if is_mask else TT.InterpolationMode.BILINEAR,
                "antialias": not is_mask
            }
            with catch_warnings(): # suppress warnings about the antialias argument
                simplefilter("ignore")
                image = TT.functional.resize(image, self.size, **resize_kw)
        tensor = TT.functional.pil_to_tensor(image)
        if not is_mask and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1) # convert grayscale to RGB by just repeating the channel 3 times
        return tensor

    def _preprocess_image(self, img: torch.Tensor) -> torch.Tensor:
        """ input batch tensor and return a tensor of the same shape """
        if self.do_rescale:
            img = TT.functional.to_dtype(img, dtype=torch.float32, scale=True)
        if self.do_normalize:
            img = TT.functional.normalize(img, mean=self.data_mean, std=self.data_std)
        return img

    def _preprocess_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.do_reduce_labels:
            # just going to reduce all labels and assume the background label is 0 - extend for specified background label later
            mask = torch.clamp(mask - 1, min=0) # reduce all labels by 1
        mask = TT.functional.to_dtype(mask, dtype=torch.int64, scale=False)
        return mask

    def _collate(self, images: List[Image], is_mask: bool = False) -> torch.Tensor:
        # convert images to tensors and stack them into a single tensor (B, C, H, W)
        images = [self._to_tensor(im, is_mask) for im in images]
        images: torch.Tensor = torch.stack(images)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True) # move to GPU if available
        prep_fn = self._preprocess_mask if is_mask else self._preprocess_image
        images = prep_fn(images)
        # # convert to long tensor for segmentation maps
        # images = images.long()
        return images

    def __call__(self, images: List[Image], segmentation_maps: Optional[List[Image]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        sample = {}
        sample["pixel_values"] = self._collate(images, is_mask=False)
        # convert segmentation maps to tensors and stack them into a single tensor (B, H, W)
        if segmentation_maps is not None:
            sample["labels"] = self._collate(segmentation_maps, is_mask=True)
        return sample



def load_model_and_processor(task: str, variant: str, strategy: Optional[str], pretrained_name: str, num_labels: int):
    """ Load a Segformer model and processor from HuggingFace. Ignore the `strategy` argument if the variant is not Lipschitz
        Args
            task : "segmentation" | "classification"
            variant : "baseline" | "lipschitz"
            strategy : geometric_mean | spectral_norm | stable_softplus | implicit_layer | jacobian_norm
    """
    from models.model_registry import get_segformer_model
    #from transformers import SegformerImageProcessor
    model = get_segformer_model(task, variant, pretrained_name, num_labels=num_labels, lipschitz_strategy=strategy)
    #model = torch.compile(model, mode="reduce-overhead")   # PyTorch 2.1+
    # use do_reduce_labels for Scene-Parse-150 since ID 0 is reserved for "other objects" and is not used in official evaluation
    # processor = (
    #     SegformerImageProcessor.from_pretrained(pretrained_name, do_reduce_labels=True, do_resize=False)
    #     if task == "segmentation" else
    #     SegformerImageProcessor.from_pretrained(pretrained_name, size={"height": 256, "width": 256})
    # )
    processor = CustomImageProcessor(
        do_reduce_labels=(task == "segmentation"),
        size=(256, 256)
    )
    return model, processor



# minimal preprocessing to convert image types to tensor and move to CUDA earlier before using the `transformers` preprocessor
# TORCH_PREPROCESSOR = {
#     "image": TT.Compose([
#         # TODO: remove hard-coding after deciding whether to keep this
#         TT.Resize((256, 256), interpolation=TT.InterpolationMode.BILINEAR, antialias=True), # resize comes first since the PIL backend is typically faster
#         TT.PILToTensor(),
#         TT.Lambda(lambda img: img.repeat(3, 1, 1) if img.shape[0] == 1 else img), # convert grayscale to RGB by just repeating the channel 3 times
#         # TT.Lambda(lambda img: img.to(device="cuda") if torch.cuda.is_available() else img),
#     ]),
#     "mask": TT.Compose([
#         TT.Resize((256, 256), interpolation=TT.InterpolationMode.NEAREST), # resize comes first since the PIL backend is typically faster
#         TT.PILToTensor(),
#         # TT.Lambda(lambda img: img.to(device="cuda") if torch.cuda.is_available() else img)
#     ])
# }

# def to_cuda(img):
#     return img.to(device="cuda") if torch.cuda.is_available() else img

# def debug_grayscale_image(images: List[Image]):
#     print("Image shapes (as numpy): ")
#     for im in images:
#         img = np.array(im)
#         if img.ndim < 3 or img.shape[2] != 3:
#             print("Image shape: ", img.shape)
#             plt.imshow(img, cmap="gray")
#             plt.show()
#             #raise ValueError("Image is not RGB or has an unexpected number of dimensions.")

# # TODO: pass preprocessing arguments and use the functional versions of the torchvision implementations
# def convert_sample_to_tensors(sample: Dict[str, Image], task="segmentation") -> Dict[str, torch.Tensor]:
#     images = [b.get("pixel_values", b.get("image", None)) for b in sample]
#     # debug_grayscale_image(images)
#     images = [TORCH_PREPROCESSOR["image"](im) for im in images]
#     print("image shapes: ", [im.shape for im in images])
#     images = to_cuda(torch.stack(images))
#     masks = [b.get("labels", b.get("label", b.get("annotation", None))) for b in sample]
#     if task == "segmentation":
#         masks = [TORCH_PREPROCESSOR["mask"](m) for m in masks]
#         masks = to_cuda(torch.stack(masks))
#     else:
#         # masks is actually a single integer class label for each batch index
#         masks = torch.tensor(masks).unsqueeze(0).transpose(1,0) # list of ints of length B -> LongTensor of shape (B, 1)
#     print("mask shapes: ", [m.shape for m in masks])
#     return {"pixel_values": images, "labels": masks}


# def segformer_collate(batch, preprocessor, task="segmentation"):
#     # batch image type: class 'PIL.JpegImagePlugin.JpegImageFile', mask type: class 'PIL.PngImagePlugin.PngImageFile'
#     sample = convert_sample_to_tensors(batch, task)
#     # stack images into a single tensor (B, C, H, W) - use a fallback for the key name in case it's different
#     # TODO: consider using tensordict library for dealing with images and labels instead
#     if task == "segmentation":
#         # TODO: I'm considering just writing a custom preprocessor since this has so many gotchas
#         #? NOTE: technically sample becomes a UserDict subclass of type `BatchFeature` here, but the internal data is a dict of tensors
#         sample = preprocessor(images = sample["pixel_values"], segmentation_maps = sample["labels"], return_tensors="pt")
#         assert "labels" in sample, "Segmentation maps not found in the preprocessed sample."
#     else:  # classification
#         ###images = [b.get("pixel_values", b.get("image", None)) for b in batch]
#         ###sample = preprocessor(images=images, return_tensors="pt")
#         ###labels = torch.tensor([b["label"] for b in batch], dtype=torch.long) # shape (B,)
#         #? NOTE: preprocessor normally returns a BatchFeature object, but the object's internal data are tensors, so querying keys gives the right values
#         sample["pixel_values"] = preprocessor(images=sample["pixel_values"], return_tensors="pt")["pixel_values"]
#         assert "labels" in sample, "Classification labels tensor not found in the preprocessed sample."
#         #labels = torch.stack([b["label"] for b in batch], dim=0, dtype=torch.long)
#         ###sample["labels"] = labels
#     #sample["labels"] = sample["labels"].to(dtype=torch.int32) # cross-entropy loss expects long, so can't use this as is
#     return sample


def segformer_collate(batch, preprocessor: CustomImageProcessor, task="segmentation"):
    sample = {}
    images = [b.get("pixel_values", b.get("image", None)) for b in batch]
    masks = [b.get("labels", b.get("label", b.get("annotation", None))) for b in batch]
    if task == "segmentation":
        sample = preprocessor(images=images, segmentation_maps=masks)
        if sample["labels"].dim() != 3:
            sample["labels"] = sample["labels"].squeeze(dim=1)
    else:  # classification
        sample = preprocessor(images=images)
        # masks is actually a single integer class label for each batch index
        sample["labels"] = torch.tensor(masks, dtype=torch.long).unsqueeze(-1) # list of ints of length B -> LongTensor of shape (B, 1)
    return sample


# https://huggingface.co/docs/transformers/en/main_classes/trainer
class HuggingFaceModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # TODO: might want to make the default True to always keep logits
        # labels = inputs.pop("labels")
        # outputs = model(**inputs, labels=labels)
        loss, outputs = None, None
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if isinstance(out, tuple):
            loss, outputs = out
        else:
            loss = out
        # TODO: may end up changing this structure so revisit later
        lip_loss = 0.0
        if hasattr(model, "get_lipschitz_loss"):
            lip_loss = model.get_lipschitz_loss() * self.args.lambda_lip
        print("\nBase loss: ", loss.item() if isinstance(loss, torch.Tensor) else loss, "\t",
              "Lipschitz loss: ", lip_loss.item() if isinstance(lip_loss, torch.Tensor) else lip_loss)
        loss += lip_loss
        return (loss, outputs) if return_outputs else loss