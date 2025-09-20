# Black Sigatoka Detection using CNN + ViT

This repository contains a CNN + ViT hybrid model for detecting Black Sigatoka disease in banana leaves. The project includes training, inference, and evaluation scripts.

---

## Introduction

Black Sigatoka, also known as Black Leaf Streak Disease, is a fungal disease caused by Mycosphaerella fijiensis that affects banana plants. It is one of the most destructive diseases of bananas worldwide, causing significant yield losses and reducing fruit quality. Early and accurate detection of Black Sigatoka is crucial for timely intervention and effective disease management.

### Existing Works

Several approaches have been explored for automatic detection of Black Sigatoka using image-based analysis:

- Traditional Machine Learning: Feature extraction methods combined with classifiers like SVM or Random Forest have been used to detect leaf spots. These approaches require careful hand-crafted feature design and are often sensitive to variations in lighting and leaf orientation.
- Convolutional Neural Networks (CNNs): CNNs have been applied to classify diseased and healthy banana leaves, achieving high accuracy while automatically learning relevant features from images.
- Vision Transformers (ViTs): ViTs leverage self-attention mechanisms and have recently shown strong performance in image classification tasks, particularly for capturing long-range dependencies and global context in leaf images.

### Our Approach

We propose a hybrid CNN + ViT model that combines the local feature extraction capabilities of CNNs with the global context modeling of Vision Transformers. This fusion allows the model to capture both fine-grained textures and overall leaf patterns effectively.

Using this hybrid approach, we achieved a test accuracy of 99.91% on the Black Sigatoka dataset.

---

## Project Structure

Black_Sigatoka/
│
├── datasets/ # Dataset folder (ignored in Git)
│ ├── train/
│ ├── val/
│ └── test/
│
├── Inference/ # Inference outputs (weights not included)
│ ├── confusion_matrix.png
│ ├── predictions.csv
│ └── sample_pred_*.png
│
├── train.py # Script for dataset preparation and model training
├── test.py # Script for running inference on images
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore file
└── README.md # This file

yaml
Copy code

---

## Dataset

The dataset used in this project is from Harvard Dataverse:

[The Nelson Mandela African Institution of Science and Technology Bananas dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LQUWXW)

Note: The dataset is not included in the repository due to size constraints. Download it and place the images in the folder structure as described above.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/YourUsername/Black_Sigatoka_CNN-ViT.git
cd Black_Sigatoka_CNN-ViT
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Dataset Preparation
Place your downloaded dataset into the root folder, for example healthy_1/ and black sigatoka_1/.

Run dataset preparation and training:

bash
Copy code
python train.py
This will:

Split the dataset into train, val, and test sets.

Train the CNN + ViT hybrid model.

Save model weights in Inference/cnn_vit_model.pth.

Inference
Run inference on a single image:

bash
Copy code
python test.py --image path/to/your/image.jpg
Predictions, confusion matrix, and sample images will be saved in the Inference/ folder.

Note: Model weights are not included in this repo due to size (~370 MB). You need to either train the model locally or provide a download link for the weights.

Results
Confusion matrix: Inference/confusion_matrix.png

Predictions CSV: Inference/predictions.csv

Sample predictions: Inference/sample_pred_*.png

Dependencies
See requirements.txt:

shell
Copy code
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.5
matplotlib>=3.7.1
pandas>=2.1.0
scikit-learn>=1.2.2
Pillow>=9.5.0
Notes
.gitignore excludes the datasets/ folder and large model weights (cnn_vit_model.pth) to keep the repository lightweight.

Ensure dataset folder names match those in train.py (healthy_1 and black sigatoka_1).

Enable long paths on Windows if you face path length issues:

powershell
Copy code
git config --global core.longpaths true

License
This project is released under the MIT License.

