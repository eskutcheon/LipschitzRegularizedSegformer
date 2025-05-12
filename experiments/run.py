
import os
import argparse
from pathlib import Path
from functools import partial
from datasets import disable_caching
disable_caching() # using streaming datasets so ensure no accidental disk writes
# local imports
import experiments.config as cfg
from experiments.loading import (
    load_streaming_dataset,
    random_streaming_split,
    compute_max_steps
)
from experiments.train import (
    HuggingFaceModelTrainer,
    get_training_args,
    build_compute_metrics_fn,
    load_model_and_processor,
    segformer_collate
)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer variants")
    parser.add_argument("--task", choices=["segmentation", "classification"], required=True, help="Specified downstream task")
    #parser.add_argument("--model") ###!!!! FIXME: add registry for baseline model or the Lipschitz MLP variant used to construct the LRSegformer classes
    #parser.add_argument("--dataset_name", default=cfg.SEGMENTATION_TRAIN_DATASET)
    # dataset arguments
    #? NOTE: defaults to datasets in config based on the task - set in main()
    parser.add_argument("--dataset", default=None, help="HF dataset name (default set by task)")
    parser.add_argument("--train_pct", type=float, default=0.8, help="Fraction for training in random split")
    parser.add_argument("--max_train", type=int, default=None, help="Fixed n samples for training split")
    parser.add_argument("--max_val", type=int, default=None, help="Fixed n samples for validation split")
    # model arguments
    parser.add_argument("--variant", choices=["baseline", "lipschitz"], default="baseline", help="Baseline or Lipschitz-regularized Segformer variants")
    parser.add_argument("--strategy", default=None, help="Lipschitz layer strategy (ignored for baseline)")
    # optimization and training arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for optimizer (default is AdamW)")
    parser.add_argument("--lambda_lip", type=float, default=0.1, help="Lipschitz regularization weight (ignored for baseline)")
    parser.add_argument("-o", "--out_dir", type=Path, default=cfg.CHECKPOINT_DIR, help="Directory for checkpoints & logs")
    return parser.parse_args()

# TODO: replace this with instantiation of a dataclass and add validation to the __post_init__
def validate_args(args):
    if args.dataset is None:
        args.dataset = (cfg.SEGMENTATION_TRAIN_DATASET if args.task == "segmentation" else cfg.CLASSIFICATION_TRAIN_DATASET)
    args.max_train = args.max_train or cfg.DEFAULT_MAX_TRAIN
    if args.variant == "baseline" and args.strategy is not None:
        print("WARNING: Lipschitz strategy should not be specified for baseline models and will be ignored.")
    elif args.variant != "baseline" and args.strategy is None:
        print("WARNING: Lipschitz strategy should be specified for Lipschitz models. Defaulting to 'geometric_mean'.")
        args.strategy = "geometric_mean"
    return args

# import sys

def save_model(trainer: HuggingFaceModelTrainer, out_dir: str, model_type: str, strategy: str = None):
    ckpt_name = f"{model_type}"
    append_counter = lambda name: "-" + str(len([f for f in os.listdir(out_dir) if name in f]) + 1)
    if model_type == "lipschitz":
        ckpt_name += f"-{strategy}"
        if ckpt_name in os.listdir(out_dir):
            ckpt_name += append_counter(ckpt_name)
    else:
        ckpt_name += append_counter("baseline")
    out_path = os.path.join(out_dir, ckpt_name)
    trainer.save_model(out_path)
    print(f"Model saved to {out_path}")



def main():
    args = validate_args(parse_cli())
    #!!! FIXME: need to refactor this logic for getting the number of classes later
    num_classes = 150 if args.task == "segmentation" else 1000
    model, processor = load_model_and_processor(
        task = args.task,
        variant = args.variant,
        strategy = args.strategy,
        pretrained_name = cfg.PRETRAINED_WEIGHTS,
        num_labels=num_classes
    )
    #! figure out a safer way to determine whether to set remote code - should couple it with the datasets in the config
        #! or possibly add it to a try-except block and ask the user whether to trust it
    ds_raw = load_streaming_dataset(args.dataset, split="train", trust_remote_code=True)
    splits = random_streaming_split(ds_raw, train_pct=args.train_pct, max_train=args.max_train, max_val=args.max_val)
    max_steps = compute_max_steps(splits, args.batch_size, args.epochs, args.max_train)
    #collate = lambda batch: segformer_collate(batch, processor, task=args.task)
    collate_fn = partial(segformer_collate, preprocessor=processor, task=args.task)
    # training steps
    training_args = get_training_args(
        output_dir=args.out_dir,
        max_steps=max_steps,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_lip=args.lambda_lip
    )
    trainer = HuggingFaceModelTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        data_collator=collate_fn,
        compute_metrics=build_compute_metrics_fn(num_classes, args.task),
    )
    print("BEGINNING TRAINING...")
    trainer.train()
    save_model(trainer, args.out_dir, args.variant, args.strategy)

if __name__ == "__main__":
    main()