# =====================================================
# CNN + ViT Model Single Image Inference
# =====================================================

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16, resnet18
from PIL import Image
import argparse

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.getcwd()
INFERENCE_DIR = os.path.join(BASE_DIR, "Inference")
WEIGHTS_PATH = os.path.join(INFERENCE_DIR, "cnn_vit_model.pth")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# ------------------------------
# Device
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# Model Definition
# ------------------------------
class CNN_ViT_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_ViT_Model, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        self.vit = vit_b_16(pretrained=False)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, 256)

        self.fusion = nn.Linear(256*2, num_classes)

    def forward(self, x, stage='fusion'):
        if stage == 'cnn':
            return self.cnn(x)
        elif stage == 'vit':
            return self.vit(x)
        else:
            cnn_feat = self.cnn(x)
            vit_feat = self.vit(x)
            fused = torch.cat([cnn_feat, vit_feat], dim=1)
            out = self.fusion(fused)
            return out

# ------------------------------
# Load model
# ------------------------------
num_classes = len(os.listdir(os.path.join(DATASET_DIR, "train")))
model = CNN_ViT_Model(num_classes=num_classes)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

# ------------------------------
# Argument parser for image path
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to the image to test")
args = parser.parse_args()
image_path = args.image

# ------------------------------
# Image preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# ------------------------------
# Inference
# ------------------------------
with torch.no_grad():
    outputs = model(img_tensor)
    pred_idx = torch.argmax(outputs, dim=1).item()
    class_names = sorted(os.listdir(os.path.join(DATASET_DIR, "train")))
    pred_class = class_names[pred_idx]

print(f"Image: {os.path.basename(image_path)} --> Predicted Class: {pred_class}")
