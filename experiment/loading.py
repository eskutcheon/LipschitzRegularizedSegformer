import torch
from datasets import load_dataset, Image, IterableDatasetDict, IterableDataset
from transformers.models.segformer import SegformerImageProcessor


def load_streaming_dataset(name: str, split: str, streaming: bool = True, **kwargs) -> IterableDataset: #torch.utils.data.IterableDataset:
    ds = load_dataset(name, split=split, streaming=streaming, **kwargs)
    # ds = ds.cast_column("image", torch.Tensor)
    ds = ds.cast_column("image", Image()) # ensures PIL -> array conversion
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
    MAX_BUFFER = num_samples or 10000 # max buffer size for shuffling
    ds = ds.shuffle(seed=seed, buffer_size=MAX_BUFFER)  # big enough for decent mix
    if max_train is None and max_val is None:
        # decide how many examples to take without len(ds)
        max_train = int(0.8 * MAX_BUFFER) # change to suit GPU/epoch budget
        max_val = int(max_train * (1 - train_pct) / train_pct)
    train_ds = ds.take(max_train)
    val_ds = ds.skip(max_train).take(max_val)
    return IterableDatasetDict(train=train_ds, validation=val_ds)


def segformer_collate(batch, preprocessor, task="segmentation"):
    sample = {}
    images = [b["image"] for b in batch]
    if task == "segmentation":
        #! FIXME: don't think this will always be the same key for all datasets
        maps = [b["annotation"] for b in batch]
        sample = preprocessor(images=images, segmentation_maps=maps, return_tensors="pt")
        #labels = sample.pop("labels") # should be shape (B, 1, H, W)
        assert "labels" in sample, "Segmentation maps not found in the preprocessed sample."
    else:  # classification
        sample = preprocessor(images=images, return_tensors="pt")
        labels = torch.tensor([b["label"] for b in batch])
        sample["labels"] = labels
    return sample


# TODO: pass `functools.partial(segformer_collate, processor=processor, task=task)` to `transformers.Trainer`