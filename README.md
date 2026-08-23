# GenAI-Image-Classification-using-CNN-and-Transfer-Learning
Practical No-01: Image Classification using CNN and Transfer Learning
# 🧬 Breast Cancer Histopathological Image Classification
## Custom CNN vs VGG16 Transfer Learning

A deep learning image-classification project that compares a **Custom Convolutional Neural Network (CNN) trained from scratch** with **VGG16 Transfer Learning** for classifying breast histopathological images as **Benign** or **Malignant**.

This project is implemented as part of **Practical Assignment 1 – Image Classification using CNN and Transfer Learning**.

---

## 📌 Project Overview

Breast cancer diagnosis often involves the examination of histopathological tissue images under a microscope. Manually analyzing these images can be challenging because tissue patterns can be complex and visually similar.

This project investigates whether **transfer learning using a pretrained CNN** can provide better classification performance than training a CNN completely from scratch.

Two approaches are implemented and evaluated on the same dataset:

1. **Custom CNN** – trained from scratch with randomly initialized weights.
2. **VGG16 Transfer Learning** – uses ImageNet-pretrained weights followed by feature extraction and fine-tuning.

The two models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Training time
- Number of parameters
- Training/validation loss
- Training/validation accuracy
- Confusion matrix

---

# 🎯 Objectives

The main objectives of this project are:

- To perform binary classification of breast histopathological images.
- To preprocess and prepare the BreaKHis dataset for deep learning.
- To develop a Custom CNN from scratch.
- To implement VGG16 using transfer learning.
- To perform feature extraction using pretrained VGG16 weights.
- To fine-tune the last convolutional block of VGG16.
- To evaluate both models using standard classification metrics.
- To compare CNN and transfer-learning approaches.
- To understand and reproduce the core methodology of the selected research paper.
- To analyze the advantages and limitations of both approaches.

---

# 📚 Research Paper

### Selected Research Paper

**Breast cancer histology images classification: Training from scratch or transfer learning?**

**Authors:** Shallu Sharma and Rajesh Mehra  
**Published:** 2018  
**Journal:** ICT Express

The research paper investigates whether deep CNN models trained from scratch or using transfer learning provide better performance for breast cancer histopathological image classification.

The paper evaluates models including:

- VGG16
- VGG19
- ResNet50

The paper reports its best result using **VGG16 transfer learning**, achieving approximately **92.60% accuracy** under its experimental setup.

### 📄 Research Paper Included

The research paper PDF is included in this repository:

```text
CNNBreast cancer.pdf
