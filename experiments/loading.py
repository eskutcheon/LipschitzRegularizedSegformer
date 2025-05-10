
from math import ceil
import torch
from datasets import load_dataset, Image, IterableDatasetDict, IterableDataset, SplitInfo
from pprint import pprint



def load_streaming_dataset(name: str, split: str, streaming: bool = True, **kwargs) -> IterableDataset: #torch.utils.data.IterableDataset:
    ds = load_dataset(name, split=split, streaming=streaming, **kwargs)
    # ds = ds.cast_column("image", torch.Tensor)
    # TODO: rename columns to be compatible with Segformer classes (Trainer automatically drops incompatible columns)
    ds = ds.cast_column("image", Image()) # ensures PIL -> array conversion
    if "image" in ds.column_names:
        ds = ds.rename_column("image", "pixel_values")
    if "annotation" in ds.column_names:
        ds = ds.rename_column("annotation", "labels")
    return ds # ds.with_format("torch")           # dict[str, Tensor]


def random_streaming_split(ds: IterableDataset,
                           num_samples: int | None = None,
                           train_pct: float = 0.8,
                           seed: int = 42,
                           max_train: int | None = None,
                           max_val: int | None = None) -> IterableDatasetDict:
    """ Turn a *single* streaming dataset into {train, validation}.
        It works even without knowing the dataset length: we shuffle once, then take / skip.
        If we consider class imbalance concerns, move to pre-defined splits instead
    """
    #buf_size = min(max_train, 1024)
    num_samples = num_samples or 10000 # max buffer size for shuffling
    #ds = ds.shuffle(seed=seed, buffer_size=buf_size)  # big enough for decent mix
    if hasattr(ds.info, "splits"):
        split_info: SplitInfo = ds.info.splits["train"]
        if hasattr(split_info, "num_examples"):
            num_samples = split_info.num_examples
    if max_train is None and max_val is None:
        # decide how many examples to take without len(ds)
        max_train = int(0.8 * num_samples) # change to suit GPU/epoch budget
        max_val = int(max_train * (1 - train_pct) / train_pct)
    train_ds = ds.take(max_train)
    # print(train_ds.info)
    # print("train_ds num samples: ", train_ds.info.num_examples)
    val_ds = ds.skip(max_train).take(max_val)
    #print("val_ds num samples: ", val_ds.info.num_examples)
    return IterableDatasetDict(train=train_ds, validation=val_ds, info={"train_len": max_train, "val_len": max_val})


def compute_max_steps(splits: IterableDatasetDict, batch_size: int, num_epochs: int, default_max: int) -> int:
    # Try to recover the exact train length from the helper
    train_examples = default_max
    if isinstance(splits, dict) and "train_len" in splits.get("info", {}):
        train_examples = splits["info"]["train_len"]
    steps_per_epoch = ceil(train_examples / batch_size)
    return steps_per_epoch * num_epochs # should probably not be multiplied but I need to see how the trainer handles the progress bar when a new epoch starts first

