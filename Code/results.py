# =====================================================
# CNN + ViT Model Inference
# =====================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_16, resnet18
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from PIL import Image

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
INFERENCE_DIR = os.path.join(BASE_DIR, "Inference")
WEIGHTS_PATH = os.path.join(INFERENCE_DIR, "cnn_vit_model.pth")

os.makedirs(INFERENCE_DIR, exist_ok=True)

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
        self.cnn = resnet18(pretrained=False)  # weights already saved
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
# Load model weights
# ------------------------------
num_classes = len(os.listdir(os.path.join(DATASET_DIR, "train")))
model = CNN_ViT_Model(num_classes=num_classes)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

# ------------------------------
# Test Dataset
# ------------------------------
test_dataset = ImageFolder(
    os.path.join(DATASET_DIR, "test"),
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------------
# Inference
# ------------------------------
y_true, y_pred, filenames = [], [], []

with torch.no_grad():
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        filenames.append(test_dataset.imgs[idx][0])

# ------------------------------
# Accuracy
# ------------------------------
accuracy = sum([y_true[i]==y_pred[i] for i in range(len(y_true))]) / len(y_true)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(INFERENCE_DIR, "confusion_matrix.png"))
plt.close()

# ------------------------------
# Save Predictions CSV
# ------------------------------
df = pd.DataFrame({
    "Filename": [os.path.basename(f) for f in filenames],
    "True": [test_dataset.classes[i] for i in y_true],
    "Predicted": [test_dataset.classes[i] for i in y_pred]
})
df.to_csv(os.path.join(INFERENCE_DIR, "predictions.csv"), index=False)

print(f"Inference completed. Confusion matrix and predictions saved in '{INFERENCE_DIR}'.")
