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
