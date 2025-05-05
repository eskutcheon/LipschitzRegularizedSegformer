
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from ..experiment.metrics import get_MetricTracker



def train_one_epoch(model, dataloader, optimizer, device, metric_tracker):
    model.train()
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        preds = outputs.logits.detach()
        metric_tracker.update(preds, labels)
        progress_bar.set_postfix(loss=loss.item())


def validate(model, dataloader, device, metric_tracker):
    model.eval()
    metric_tracker.reset()
    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            preds = outputs.logits
            metric_tracker.update(preds, labels)
            progress_bar.set_postfix(loss=loss.item())
    metrics_result = metric_tracker.compute()
    return metrics_result


def train(model, train_loader, val_loader, epochs, device, optimizer, metric_classes, num_classes):
    metric_tracker = get_MetricTracker(metric_classes, num_classes, device)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_one_epoch(model, train_loader, optimizer, device, metric_tracker)
        val_metrics = validate(model, val_loader, device, metric_tracker)
        print(f"Validation Metrics: {val_metrics}")
