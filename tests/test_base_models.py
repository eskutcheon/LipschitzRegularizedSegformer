
import sys
import torch
import torchvision.transforms.v2 as TT
from torchvision.datasets import VOCSegmentation, CocoDetection
from torcheval.metrics import MulticlassAccuracy
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerForImageClassification,
    AutoImageProcessor
)
from datasets import load_dataset
from PIL import Image


# ---------- Set device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------- Load Models ----------
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=20).to(device)
cls_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0", num_labels=1000).to(device)
# ---------- Image Processor ----------
processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", use_fast=True)
# ---------- Transforms ----------
img_transform = TT.Compose([
    TT.Resize((512, 512), interpolation=TT.InterpolationMode.BILINEAR),
    TT.ToTensor(),
    TT.ToDtype(torch.float32, scale=True),
])
mask_transform = TT.Compose([
    TT.Resize((512, 512), interpolation=TT.InterpolationMode.NEAREST),
    TT.ToTensor(),
    TT.ToDtype(torch.int64, scale=False),
    #TT.Lambda(lambda x: x.unsqueeze(0)),
])

print("number of labels in the model: ", seg_model.config.num_labels)



def train_segmentation_model():
    # Placeholder for training logic
    # ---------- 1. Pascal VOC (Segmentation) ----------
    voc = VOCSegmentation(root="./data", year="2012", image_set="val", transform=img_transform)
    print("number of samples in VOC dataset:", len(voc))
    for test_i in range(5):
        image, mask = voc[test_i]
        print("(before preprocessing) image type: ", type(image), "mask type: ", type(mask))
        mask = mask_transform(mask)
        print("largest mask value present: ", torch.max(mask))
        print("(before preprocessing) image shape: ", image.shape, "mask shape: ", mask.shape)
        inputs = processor(images=image, return_tensors="pt").to(device)
        print("(after preprocessing) image type: ", type(image), "mask type: ", type(mask))
        print("(after preprocessing) image shape: ", inputs['pixel_values'].shape)
        with torch.no_grad():
            out = seg_model(**inputs)
        print("output fields: ", vars(out).keys())
        print("VOC Segmentation output logits: ", out.logits.shape)
        print("VOC Segmentation logits values: ", torch.min(out.logits), torch.max(out.logits))
        num_classes = out.logits.shape[1]
        print("number of classes in VOC dataset:", num_classes)
        confusion_metrics = MulticlassAccuracy(average='macro', num_classes=num_classes) #.to(device)
        logits = TT.functional.resize(out.logits, size=(512, 512), interpolation=TT.InterpolationMode.BILINEAR).cpu()
        preds = torch.softmax(logits, dim=1).argmax(dim=1).flatten(start_dim=0)
        print("shape of predictions: ", preds.shape)
        confusion_metrics.update(preds, mask.flatten(start_dim=0))
        print("VOC Segmentation confusion matrix: ", confusion_metrics.compute())
    # ---------- 2. COCO (Segmentation) ----------
    # Using dummy image from VOC for demonstration since COCO annotations are nontrivial for masks
    # print("Reusing VOC image for COCO shape test")
    # inputs = processor(images=image, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     out = seg_model(**inputs)
    # print("COCO Segmentation Logits:", out.logits.shape)


def train_classification_model():
    NUM_ITER = 10
    # ---------- 3. ImageNet-1k (HuggingFace) ----------
    # Load only 1 sample from ImageNet validation set
    imagenet_ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    print("dataset object keys: ", vars(imagenet_ds).keys())
    # ---------- 4. PACS (Classification, HuggingFace) ----------
    #pacs = load_dataset("pacs", split="validation[:1]")
    #print("number of samples in VOC dataset:", len(imagenet_ds))
    loop_count = 0
    for sample in imagenet_ds:
        if loop_count >= NUM_ITER:
            break
        #print("sample keys: ", sample.keys()) # keys: "image" and "label"
        #print("sample keys: ", imagenet_ds[test_i].keys())
        #img = imagenet_ds[test_i]["image"]
        img: Image = sample["image"]
        gt: int = sample["label"]
        print("(before preprocessing) image type: ", type(img))
        print("(before preprocessing) image shape: ", img_transform(img).shape)
        inputs = processor(images=img, return_tensors="pt").to(device)
        print("(after preprocessing) image type: ", type(inputs))
        print("batch feature object keys: ", inputs.keys())
        print("(after preprocessing) image shape: ", [v.shape for v in inputs.data.values()])
        with torch.no_grad():
            out = cls_model(**inputs)
        print("output type: ", type(out))
        print("output fields: ", vars(out).keys())
        print("ImageNet-1k Classification Logits shape: ", out.logits.shape)
        pred = torch.softmax(out.logits, dim=1).argmax()
        print("final prediction: ", pred.item(), "ground truth: ", gt)
        # hf_img = pacs[0]["image"]
        # hf_input = processor(images=hf_img, return_tensors="pt").to(device)
        # with torch.no_grad():
        #     logits = cls_model(**hf_input).logits
        # print("PACS Classification Logits:", logits.shape)
        print()
        loop_count += 1


#train_segmentation_model()
train_classification_model()