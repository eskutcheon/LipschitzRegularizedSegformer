from transformers import SegformerImageProcessor
import torch
import torch.nn.functional as F
from PIL import Image


def load_trained_model(model_cls, pretrained_name, checkpoint_path, device):
    model = model_cls.from_pretrained(pretrained_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_segmentation(model, processor, image_path, device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        predicted_segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return predicted_segmentation


def predict_classification(model, processor, image_path, device, id2label):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = logits.argmax(dim=1).item()
    return id2label[predicted_class]
