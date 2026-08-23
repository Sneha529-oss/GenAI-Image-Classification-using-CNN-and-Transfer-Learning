# Generative AI - Practical No-01: Image Classification using CNN and Transfer Learning

# Breast Cancer Histopathological Image Classification — CNN vs Transfer Learning (VGG16)

Image classification pipeline built for a Generative AI lab assignment, comparing a **Custom CNN trained from scratch** against **VGG16 transfer learning**, on histopathological breast cancer images. Implementation is grounded in and compared against a peer-reviewed research paper on the same problem.

---

## 📋 Project Info

| Field | Details |
|---|---|
| **Student** | Sneha Chaurasia |
| **PRN** | 202401110046 |
| **Class / Division** | CSE-AIML / A |
| **Subject** | Generative AI Lab |
| **Practical No.** | 01 |

---

## 📄 Research Paper

| Field | Details |
|---|---|
| **Title** | Breast cancer histology images classification: Training from scratch or transfer learning? |
| **Authors** | Shallu Sharma, Rajesh Mehra |
| **Journal** | ICT Express (2018) |
| **DOI** | [10.1016/j.icte.2018.10.007](https://doi.org/10.1016/j.icte.2018.10.007) |
| **Link** | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405959518304934) |

**Problem Statement:** Manual histopathological analysis by pathologists is time-consuming and prone to inter-observer variability. CNNs can automatically learn discriminative features from tissue images, but training deep CNNs from scratch is data-hungry — this raises the question of whether transfer learning from ImageNet-pretrained models is more effective for this domain.

The paper compared VGG16, VGG19, and ResNet50 as fixed feature extractors and fine-tuned models (with logistic regression/SVM heads) on the full multi-magnification BreaKHis dataset, reporting a best VGG16 transfer-learning accuracy of **92.60%** on a 90–10 split.

---

## 📦 Dataset

| Field | Details |
|---|---|
| **Name** | BreaKHis 400X (partial subset, 400X magnification only) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/forderation/breakhis-400x) |
| **Classes** | Benign, Malignant |
| **Original dataset** | BreaKHis — collected with P&D Laboratory, Brazil; 4 magnifications (40X/100X/200X/400X) |

**Split-wise image counts (auto-counted from the dataset, not hardcoded):**

| Split | Benign | Malignant | Total |
|---|---|---|---|
| Train | 371 | 777 | 1148 |
| Test | 176 | 369 | 545 |

**Training pipeline split** (train directory further split 80/20 for validation):

| Split | Samples |
|---|---|
| Training | 919 |
| Validation | 229 |
| Testing (held out) | 545 |

---

## 🧠 Part 1: Custom CNN (from scratch)

- **Architecture:** 4 convolutional blocks + batch normalization + global average pooling + dense classification head
- **Input size:** 224 × 224 × 3
- **Preprocessing:** `rescale=1./255` (simple 0–1 normalization)
- **Augmentation (train only):** horizontal flip, ±15° rotation, ±10% zoom, ±10% width/height shift
- **Training:** `EarlyStopping` (restores best weights) + `ReduceLROnPlateau`, up to 10 epochs

**Parameters:**

| Metric | Value |
|---|---|
| Total Parameters | 423,361 |
| Trainable Parameters | 422,401 |
| Non-trainable Parameters | 960 |

**Test Set Results:**

| Metric | Value |
|---|---|
| Accuracy | 0.6899 |
| Precision | 0.6931 |
| Recall | 0.9729 |
| F1-score | 0.8095 |
| ROC-AUC | 0.6167 |
| Training Time | 160.80 s |

**Confusion Matrix:**

| | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 17 (TN) | 159 (FP) |
| **Actual Malignant** | 10 (FN) | 359 (TP) |

The Custom CNN struggles badly on Benign recall (only 17/176 correctly identified) — it's heavily biased toward predicting Malignant, likely due to class imbalance (777 vs 371 in training) and limited capacity to learn discriminative features from scratch on a small dataset.

---

## 🔁 Part 2: Transfer Learning (VGG16)

- **Pre-trained model:** VGG16 (Simonyan & Zisserman, 2014), pretrained on ImageNet (1.4M images, 1000 classes)
- **Why VGG16:** it achieved the paper's best reported result (92.60%), making it the natural choice for direct comparison
- **Preprocessing:** VGG16-specific `preprocess_input` (ImageNet mean-subtraction)

### Phase 1 — Feature Extraction
Convolutional base **frozen** (ImageNet weights untouched); only the new classification head is trained.

### Phase 2 — Fine-Tuning
Last convolutional block (`block5`) unfrozen; earlier layers stay frozen. Very small learning rate (**1e-5**) used to avoid destroying pretrained weights.

| Phase | Frozen Layers | Trainable Layers |
|---|---|---|
| Before fine-tuning | 19 | 0 |
| After fine-tuning | 15 | 4 |

**Parameters:**

| Metric | Value |
|---|---|
| Total Parameters | 14,780,481 |
| Trainable Parameters | 65,793 |
| Non-trainable Parameters | 14,714,688 |

**Test Set Results:**

| Metric | Value |
|---|---|
| Accuracy | 0.8917 |
| Precision | 0.9058 |
| Recall | 0.9377 |
| F1-score | 0.9214 |
| ROC-AUC | 0.9429 |
| Training Time | 284.84 s (feature extraction + fine-tuning) |

**Confusion Matrix:**

| | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 140 (TN) | 36 (FP) |
| **Actual Malignant** | 23 (FN) | 346 (TP) |

### Feature Map Visualization
Intermediate activations visualized at three depths for a sample test image:
- **Early** (`block1_conv2`): edges and textures
- **Middle** (`block3_conv3`): patterns and shapes
- **Deep** (`block5_conv3`): abstract, high-level representations

---

## ⚖️ Comparative Study — Custom CNN vs VGG16

| Metric | Custom CNN | VGG16 Transfer Learning |
|---|---|---|
| **Accuracy** | 0.6899 | **0.8917** |
| **Precision** | 0.6931 | **0.9058** |
| **Recall** | **0.9729** | 0.9377 |
| **F1-score** | 0.8095 | **0.9214** |
| **ROC-AUC** | 0.6167 | **0.9429** |
| **Training Time** | **160.80 s** (faster) | 284.84 s |
| **Total Parameters** | **423,361** (smaller) | 14,780,481 |
| **Trainable Parameters** | 422,401 | **65,793** (far fewer, thanks to frozen base) |

🏆 **Higher Test Accuracy:** VGG16 Transfer Learning
🏆 **Higher F1-score:** VGG16 Transfer Learning

### Advantages & Limitations

| | Custom CNN | VGG16 Transfer Learning |
|---|---|---|
| **Advantages** | Lightweight, trains faster, full architectural control | Leverages rich ImageNet features, much higher accuracy/F1/ROC-AUC, better Benign detection |
| **Limitations** | Poor Benign recall, heavily biased toward majority class, limited by small dataset size | Larger model (56 MB), longer training time, still not clinically validated |

### Conclusion

Based on the experimental results, **VGG16 Transfer Learning outperformed the Custom CNN** on every classification metric — accuracy (0.8917 vs 0.6899), F1-score (0.9214 vs 0.8095), and ROC-AUC (0.9429 vs 0.6167) — despite taking longer to train (284.84s vs 160.80s) and carrying far more total parameters (14.78M vs 423K).

This result is consistent with the direction of the reference paper's findings: transfer learning from ImageNet-pretrained weights provides a strong inductive bias that a small custom CNN cannot match when trained from scratch on a limited histopathology dataset.

**Note:** These findings are based on the smaller BreaKHis 400X subset used here and do **not** constitute a clinically validated diagnostic result.

---

## 📚 How This Implementation Relates to the Research Paper

| | Research Paper | This Implementation |
|---|---|---|
| **Dataset** | Full BreaKHis (all 4 magnifications) | BreaKHis 400X subset only |
| **Models compared** | VGG16, VGG19, ResNet50 (as feature extractors + fine-tuned) | Custom CNN (from scratch) vs VGG16 (feature extraction + fine-tuning) |
| **Classifier head** | Logistic Regression / SVM on extracted features | Dense neural network head (end-to-end trainable) |
| **Split** | 90% train / 10% test | Kaggle-provided train/test split, 80/20 train/validation within train |
| **Best reported result** | VGG16 transfer learning: 92.60% | VGG16 transfer learning: 89.17% |

**Key takeaway from the paper:** transfer learning consistently outperforms training from scratch on histopathological image classification, particularly when labeled medical imaging data is limited — a finding this implementation independently reproduces at smaller scale.

---

## 📁 Repository Structure

```
breast-cancer-cnn-transfer-learning/
│
├── README.md                                                  ← this file
├── GenAI_Image_ClassificationusingCNN_TransferLearning.ipynb   ← main notebook (executed)
├── research_paper.pdf                                          ← Sharma & Mehra (2018), ICT Express
│
└── dataset/
    ├── train/
    │   ├── benign/
    │   └── malignant/
    └── test/
        ├── benign/
        └── malignant/
```

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd breast-cancer-cnn-transfer-learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the `dataset/` folder (with `train/` and `test/` subfolders as shown above) sits alongside the notebook — or update the dataset path cell if using Kaggle's `kagglehub` download directly.
4. Run all cells:
   ```bash
   jupyter notebook GenAI_Image_ClassificationusingCNN_TransferLearning.ipynb
   ```
   A GPU runtime is strongly recommended (VGG16 fine-tuning took ~285s on GPU).

---

## 📦 Requirements

```
tensorflow
numpy
pandas
matplotlib
scikit-learn
kagglehub
jupyter
```

---

## ✅ Submission Checklist

- [x] Code file (Jupyter Notebook, executed end-to-end)
- [x] Dataset folder / dataset link included above
- [x] Research paper PDF
- [x] CNN architecture, training, and evaluation
- [x] Transfer learning (feature extraction + fine-tuning)
- [x] README file

---

## 📖 References

1. Sharma, S., & Mehra, R. (2018). *Breast cancer histology images classification: Training from scratch or transfer learning?* ICT Express. https://doi.org/10.1016/j.icte.2018.10.007
2. BreaKHis 400X Dataset (Kaggle): https://www.kaggle.com/datasets/forderation/breakhis-400x
3. Spanhol, F. A., Oliveira, L. S., Petitjean, C., & Heutte, L. (2016). *A Dataset for Breast Cancer Histopathological Image Classification.*
4. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG16).

---

## 📝 Declaration

I, **Sneha Chaurasia**, confirm that this implementation was prepared for Practical Assignment 1 and that the results presented are generated from the experiments performed in this notebook.

**GitHub Repository Link:** *(insert your repo link here)*
