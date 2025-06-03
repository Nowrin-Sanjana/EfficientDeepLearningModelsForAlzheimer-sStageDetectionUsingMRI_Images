# Efficient Deep Learning Models for Alzheimer's Stage Detection Using MRI Images

## 📌 Overview
This repository contains the implementation and documentation of deep learning models designed to classify the stages of Alzheimer's disease using MRI images. The project explores the effectiveness of pre-trained convolutional neural networks (CNNs) such as VGG16, ResNet50, and EfficientNetB0 in detecting and categorizing neurodegenerative disease severity.

## 🧠 Problem Statement
Neurodegenerative diseases, particularly Alzheimer's, are a growing public health concern with limited treatment options. Early detection is challenging due to subtle and gradual symptom progression. This project leverages deep learning to analyze MRI images and classify Alzheimer's disease into four stages: **NonDemented**, **VeryMildDemented**, **MildDemented**, and **ModerateDemented**.

## 🛠️ Key Features
- **Dataset**: Utilizes a Kaggle dataset of ~44,000 skull-stripped MRI images
- **Models**: Implements and compares three CNN architectures
- **Training**: Includes data augmentation and transfer learning techniques
- **Performance Metrics**: Evaluates models using accuracy, precision, recall, and F1-score

## 📊 Results
| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| VGG16       | 78.5%    | 70.64%    | 89.88% | 79.10%   |
| ResNet50    | 95.5%    | 94.82%    | 96.80% | 95.80%   |
| EfficientNetB0 | **95.6%** | **94.44%** | **97.97%** | **96.17%** |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.0+
- Keras

### Installation
```bash
git clone https://github.com/your-username/alzheimers-detection.git
cd alzheimers-detection
pip install -r requirements.txt
```

## Dataset
The dataset was taken from [Kaggle]([https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images))) titled 'Alzheimer's Disease Multiclass Images Dataset' containing approximately 44,000 MRI images categorized into four classes:

| Class              | Image Count |
|--------------------|-------------|
| NonDemented        | 12,800      |
| VeryMildDemented   | 11,200      |
| MildDemented       | 10,000      |
| ModerateDemented   | 10,000      |

All images are:
- Skull-stripped
- Cleaned of non-brain tissue
- Standardized as .JPG files
- Preprocessed through resizing and normalization

## Model Architectures
### VGG16
- 16-layer CNN with 3x3 convolutional kernels
- Adam optimizer (lr=1e-5)
- Baseline architecture for comparison

### ResNet50
- 50-layer deep residual network
- Solves vanishing gradient problem
- More complex feature learning

### EfficientNetB0
- Lightweight compound-scaled model
- Balances depth/width/resolution
- Most efficient architecture tested

## Training Procedure
```python
# Configuration
batch_size = 32
epochs = 25
loss_function = 'categorical_crossentropy'
optimizer = 'adam'
```

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

## 📂 Repository Structure
```
alzheimers-detection/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── config.py
├── results/
├── README.md
└── requirements.txt
```
## 🔮 Future Work
- [ ] **Implement ensemble learning** - Combine predictions from multiple models to improve accuracy
- [ ] **Explore 3D CNNs** - Adapt architecture for volumetric MRI data analysis
- [ ] **Address data imbalance** - Apply techniques like:
  - Class weighting
  - Oversampling (SMOTE)
  - Advanced augmentation

## 👥 Contributors
- Nowrin Sanjana
- Fariha Zaman
- Ahanaf Abid
