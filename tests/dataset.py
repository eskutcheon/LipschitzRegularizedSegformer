import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import SegformerImageProcessor





# TODO: planning to switch to just streaming datasets to avoid using the really big ones locally.
class HuggingFaceSegformerDataset(Dataset):
    def __init__(self, hf_dataset_name: str, split: str, processor: SegformerImageProcessor, task='segmentation'):
        self.dataset = load_dataset(hf_dataset_name, split=split)
        self.processor = processor
        self.task = task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        if self.task == 'segmentation':
            segmentation_map = sample['annotation']
            processed = self.processor(images=image, segmentation_maps=segmentation_map, return_tensors="pt")
            return {
                'pixel_values': processed['pixel_values'].squeeze(),
                'labels': processed['labels'].squeeze()
            }
        elif self.task == 'classification':
            label = sample['label']
            processed = self.processor(images=image, return_tensors="pt")
            return {
                'pixel_values': processed['pixel_values'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


def get_dataloader(hf_dataset_name, processor, split='train', task='segmentation', batch_size=8, shuffle=True):
    dataset = HuggingFaceSegformerDataset(hf_dataset_name, split, processor, task)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
