import os
import pandas as pd
from PIL import Image
import numpy as np
from PIL import ImageFile

import json
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoProcessor, AutoConfig
import matplotlib.pyplot as plt

train_csv = '/home/jupyter-nafisha/normal-abnormal-chexagent-attention-loss/CSV/train.csv'
val_csv = '/home/jupyter-nafisha/normal-abnormal-chexagent-attention-loss/CSV/val.csv'
img_dir = ''

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


def bbox_to_attention(json_path, image_width, image_height, grid_size=32):

    if json_path is None:
        return torch.zeros(grid_size * grid_size), False

    with open(json_path, "r") as f:
        json_data = json.load(f)

    objects = json_data['objects']

    if len(objects) == 0:
        return torch.zeros(grid_size * grid_size), False

    attn = torch.zeros(grid_size * grid_size)

    for ann in objects:

        x_min, y_min = ann["points"]["exterior"][0]
        x_max, y_max = ann["points"]["exterior"][1]

        # Normalize 
        x1 = x_min / image_width
        y1 = y_min / image_height
        x2 = x_max / image_width
        y2 = y_max / image_height

        px1 = int(x1 * grid_size)
        px2 = int(x2 * grid_size)
        py1 = int(y1 * grid_size)
        py2 = int(y2 * grid_size)

        px1 = max(px1, 0)
        py1 = max(py1, 0)
        px2 = min(px2, grid_size - 1)
        py2 = min(py2, grid_size - 1)

        # print(px1, py1, px2, py2)  # now meaningful

        for py in range(py1, py2 + 1):
            for px in range(px1, px2 + 1):
                idx = py * grid_size + px
                attn[idx] = 1.0

    attn = attn / (attn.sum() + 1e-6)

    return attn, True


class CXR_dataset(Dataset):
    def __init__(self, csv_path, img_dir, transform= None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_id = row["image_id"]
        json_id = row['json_path']
        img_path = os.path.join(self.img_dir, image_id)
        json_path = (
            None if pd.isna(json_id)
            else os.path.join(self.img_dir, json_id)
        )

        # print(img_path)
        # print(json_path)

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        
        if self.transform:
            img = self.transform(img) 
        inputs = processor(
            images=img,
            return_tensors="pt"
        )

        pixel_values = inputs['pixel_values'].squeeze(0)   
        attn_gt, has_bbox = bbox_to_attention(json_path, width, height)

        label = 0 if row['label'] == 'Normal' else 1
        label = torch.tensor(label, dtype=torch.long)
            
        return pixel_values, label, attn_gt, has_bbox

# Datasets
train_dataset = CXR_dataset(
    train_csv,
    img_dir,
    # train_transforms
)
val_dataset = CXR_dataset(
    val_csv,
    img_dir,
    # train_transforms
)