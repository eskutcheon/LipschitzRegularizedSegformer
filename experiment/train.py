import torch
import numpy as np
from typing import Dict, Any
from transformers import Trainer, TrainingArguments, TrainerCallback, EvalPrediction



def get_training_args(output_dir, num_epochs=30, learning_rate=5e-5, batch_size=8, lambda_lip=0.1):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        dataloader_num_workers=4,
        learning_rate=learning_rate,
        #fp16=True,
        # logging_steps=10,
        # save_steps=1000,
        # evaluation_strategy="steps",
        # eval_steps=1000,
        save_total_limit=1,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # report_to="tensorboard",
        lambda_lip=lambda_lip # used within HuggingFaceModelTrainer.compute_loss() as self.args.lambda_lip
    )


# # TODO: set in the trainer with `add_callback` or pass it to the TrainingArguments instance
class MetricTrackerCallback(TrainerCallback):
    def __init__(self, metric_names, num_classes, device):
        from experiment.metrics import get_MetricTracker
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
    import experiment.config as cfg
    tracker = get_MetricTracker(cfg.ALL_METRICS, num_labels, device="cpu")
    def _fn(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, labels = eval_pred
        with torch.no_grad():
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




# https://huggingface.co/docs/transformers/en/main_classes/trainer
class HuggingFaceModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        # TODO: may end up changing this structure so revisit later
        lip_loss = model.get_lipschitz_loss()
        loss = outputs.loss + self.args.lambda_lip * lip_loss
        return (loss, outputs) if return_outputs else loss