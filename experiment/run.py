

import argparse, os
from transformers import SegformerImageProcessor
# local imports
import experiment.config as cfg
from experiment.loading import load_streaming_dataset, random_streaming_split, segformer_collate
from experiment.train import HuggingFaceModelTrainer, get_training_args, build_compute_metrics_fn
from ..models.segformer import LRSegformerForSegmentation, LRSegformerForClassification





def get_model_and_processor(task, pretrained_name, num_classes=None):
    assert task in ['segmentation', 'classification'], "Task must be either 'segmentation' or 'classification'"
    from ..models.segformer import LRSegformerForSegmentation, LRSegformerForClassification
    from transformers import SegformerImageProcessor
    if task == 'segmentation':
        model = LRSegformerForSegmentation.from_pretrained(pretrained_name, num_labels=num_classes)
    elif task == 'classification':
        model = LRSegformerForClassification.from_pretrained(pretrained_name, num_labels=num_classes)
    processor = SegformerImageProcessor.from_pretrained(pretrained_name)
    return model, processor


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["segmentation", "classification"], required=True)
    parser.add_argument("--model") ###!!!! FIXME: add registry for baseline model or the Lipschitz MLP variant used to construct the LRSegformer classes
    parser.add_argument("--dataset_name", default=cfg.SEGMENTATION_TRAIN_DATASET)
    parser.add_argument("--output_dir", default=cfg.CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val",   type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_cli()
    num_classes = 150 if args.task == "segmentation" else 1000
    model, processor = get_model_and_processor(args.task, cfg.PRETRAINED_WEIGHTS, num_classes=num_classes)
    # data
    ds_raw  = load_streaming_dataset(args.dataset_name, split="train")
    splits  = random_streaming_split(ds_raw, train_pct=0.8, max_train=args.max_train, max_val=args.max_val)
    collate = lambda batch: segformer_collate(batch, processor, task=args.task)
    # training steps
    training_args = get_training_args(
                        output_dir=args.output_dir,
                        num_epochs=args.epochs,
                        batch_size=args.batch_size)
    trainer = HuggingFaceModelTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        data_collator=collate,
        compute_metrics=build_compute_metrics_fn(num_classes, args.task),
        # callbacks=[train.MetricTrackerCallback(config.ALL_METRICS,
        #                                        num_labels,
        #                                        device="cpu")]
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main()