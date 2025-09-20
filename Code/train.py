# =====================================================
# CNN + ViT Training and Inference (Local)
# =====================================================

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_16, resnet18

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
INFERENCE_DIR = os.path.join(BASE_DIR, "Inference")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(INFERENCE_DIR, exist_ok=True)

# ------------------------------
# Device
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# Dataset Splitter
# ------------------------------
def prepare_datasets():
    source_dirs = {
        "healthy_1": os.path.join(BASE_DIR, "healthy_1"),
        "black sigatoka_1": os.path.join(BASE_DIR, "black sigatoka_1"),
    }

    splits = ["train", "val", "test"]
    split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

    for split in splits:
        for cls in source_dirs.keys():
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

    for cls, src in source_dirs.items():
        images = os.listdir(src)
        np.random.shuffle(images)

        n_total = len(images)
        n_train = int(split_ratio["train"] * n_total)
        n_val = int(split_ratio["val"] * n_total)

        split_files = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:],
        }

        for split, files in split_files.items():
            for f in files:
                shutil.copy(
                    os.path.join(src, f),
                    os.path.join(DATASET_DIR, split, cls, f)
                )
    print("Dataset prepared with train/val/test splits.")


# ------------------------------
# CNN + ViT Model
# ------------------------------
class CNN_ViT_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_ViT_Model, self).__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        self.vit = vit_b_16(pretrained=True)
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
# Training
# ------------------------------
def train_model(epochs_stage=[6,6,6], batch_size=16, lr=1e-4):
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNN_ViT_Model(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stages = ['cnn', 'vit', 'fusion']
    for stage_idx, stage_name in enumerate(stages):
        print(f"--- Training stage {stage_idx+1}: {stage_name} ---")
        for epoch in range(epochs_stage[stage_idx]):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, stage=stage_name)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"Epoch [{epoch+1}/{epochs_stage[stage_idx]}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    # Save weights
    weights_path = os.path.join(INFERENCE_DIR, "cnn_vit_model.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved at: {weights_path}")

    print("Training completed.")
    return model


# ------------------------------
# Inference
# ------------------------------
def run_inference(model):
    model.eval()
    model.to(device)

    test_dir = os.path.join(DATASET_DIR, "test")
    test_dataset = ImageFolder(test_dir, transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    y_true, y_pred, filenames = [], [], []

    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        filenames.append(test_dataset.imgs[idx][0])

        # Save sample images
        if idx < 5:
            img = Image.open(test_dataset.imgs[idx][0]).resize((224,224))
            plt.imshow(img)
            class_names = test_dataset.classes
            plt.title(f"True: {class_names[labels.item()]}, Pred: {class_names[preds[0]]}")
            plt.axis("off")
            plt.savefig(os.path.join(INFERENCE_DIR, f"sample_pred_{idx+1}.png"))
            plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(INFERENCE_DIR, "confusion_matrix.png"))
    plt.close()

    # Save predictions CSV
    df = pd.DataFrame({
        "Filename": [os.path.basename(f) for f in filenames],
        "True": [test_dataset.classes[i] for i in y_true],
        "Predicted": [test_dataset.classes[i] for i in y_pred]
    })
    df.to_csv(os.path.join(INFERENCE_DIR, "predictions.csv"), index=False)

    print(f"Inference completed. Results saved in '{INFERENCE_DIR}'.")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    print("Preparing datasets...")
    prepare_datasets()
    print("Training CNN+ViT model...")
    model = train_model(epochs_stage=[10,10,10])  # adjust epochs per stage here
    print("Running inference...")
    run_inference(model)
    print("All results saved in 'Inference' folder.")
