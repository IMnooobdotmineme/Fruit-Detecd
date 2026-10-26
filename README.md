# 🍃 FruitFresh AI
### Fruit and Vegetable Freshness Classification Using Deep Learning

**Deep Learning Final Project — Kirirom Institute of Technology (KIT)**

**Developer:** RO ARTHIPHU

**Presentation:** [View FruitFresh AI Presentation on Canva](https://canva.link/bso3jl09ia1x3im)

---

## 1. Project Overview

FruitFresh AI is a deep learning project designed to identify fruits and vegetables and classify their visible condition as **fresh or rotten** using image classification.

The project was developed using Python, TensorFlow/Keras, and Streamlit. Users can upload an image or take a photo using their camera to receive a prediction of the produce type and freshness condition.

The project also compares a **Basic CNN trained from scratch** with **MobileNetV2 using transfer learning**, and includes a fine-tuned MobileNetV2 model.

### Main Features

- Fruit and vegetable image classification
- Fresh and rotten condition prediction
- Image upload and camera input
- Prediction scores and top five categories
- MobileNetV2 Version 1 vs Version 2 comparison
- Analytics dashboard
- Prediction history with CSV download
- Downloadable inspection reports

**Note:** The Basic CNN is currently a separate training and evaluation experiment. The website's model-comparison page compares the two MobileNetV2 versions.

---

## 2. Supported Fruits and Vegetables

The system supports **9 types of produce and 18 classification categories**.

| Produce Type | Fresh Category | Rotten Category |
|---|---|---|
| Apple | `freshapples` | `rottenapples` |
| Banana | `freshbanana` | `rottenbanana` |
| Bitter Gourd | `freshbittergroud` | `rottenbittergroud` |
| Capsicum | `freshcapsicum` | `rottencapsicum` |
| Cucumber | `freshcucumber` | `rottencucumber` |
| Okra | `freshokra` | `rottenokra` |
| Orange | `freshoranges` | `rottenoranges` |
| Potato | `freshpotato` | `rottenpotato` |
| Tomato | `freshtomato` | `rottentomato` |

The spelling `bittergroud` is retained to match the original model class labels.

---

## 3. Dataset and Preparation

### Original Dataset

**Source:** [Fresh and Stale Classification — Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)

The original dataset contained images of fresh and rotten fruits and vegetables.

| Description | Images |
|---|---:|
| Original image records | 23,619 |
| Exact duplicates removed | 3,247 |
| Unique images | 20,372 |

Exact duplicate images were identified using SHA-256 hashing and removed before splitting the dataset.

### Original Dataset Split

| Dataset | Number of Images |
|---|---:|
| Training | 16,297 |
| Validation | 2,037 |
| Testing | 2,038 |
| **Total** | **20,372** |

### Image Preprocessing

- Convert images to RGB format.
- Resize images to 224 × 224 pixels.
- Assign integer labels to the 18 categories.
- Apply model-specific input preprocessing.
- Organize training, validation, and testing datasets.

The Basic CNN uses input pixel normalization to the range 0–1. The saved MobileNetV2 models use their own preprocessing pipeline.

### Additional Dataset for Version 2

**Source:** [Fruits and Vegetables Dataset — Kaggle](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)

Additional fresh and rotten apple and tomato images were used for fine-tuning MobileNetV2.

Version 2's training dataset contained **22,271 image records**, including 1,899 additional training images.

---

## 4. Deep Learning Models

The project includes three trained model versions.

### Model 1 — MobileNetV2 Version 1

**Approach:** Transfer Learning

MobileNetV2 was initialized with pretrained ImageNet weights. The feature extractor was initially frozen, while the classification head was trained to recognize the 18 produce categories.

**Recorded Results:**

| Metric | Result |
|---|---:|
| Training Accuracy | 97.63% |
| Validation Accuracy | 97.89% |
| Original Test Accuracy | 97.55% |

These are initial results from the original prepared dataset split.

### Model 2 — MobileNetV2 Version 2

**Approach:** Fine-Tuning

During practical testing, Version 1 sometimes confused rotten tomatoes with rotten apples.

Version 2 was developed by adding apple and tomato training images and fine-tuning later MobileNetV2 layers.

| Metric | Result |
|---|---:|
| Final Training Accuracy | 99.31% |
| Final Validation Accuracy | 96.15% |
| Validation Categories | 4 |
| Training Image Records | 22,271 |

The validation results were measured on fresh and rotten apple/tomato categories only.

Version 2 corrected a previously misclassified rotten tomato in one practical test. However, its overall performance across all 18 categories has not yet been independently verified.

### Model 3 — Basic CNN

**Approach:** CNN trained from scratch

A custom Convolutional Neural Network was developed to compare training from scratch against transfer learning.

The Basic CNN uses:

- Four convolutional layers with 32, 64, 128, and 128 filters
- ReLU activation
- Max-pooling layers
- Global Average Pooling
- Dense layer with 128 units
- Dropout of 0.4
- Final Softmax layer with 18 outputs

**Training Configuration:**

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Cross-Entropy |
| Maximum Epochs | 10 |
| Output Categories | 18 |

---

## 5. CNN Model Comparison

The Basic CNN and MobileNetV2 Version 1 were evaluated on the same **2,038 images** in a subsequent exploratory comparison.

| Metric | Basic CNN | MobileNetV2 V1 |
|---|---:|---:|
| Test Accuracy | 90.68% | 98.48% |
| Macro Precision | 82.17% | 97.33% |
| Macro Recall | 84.10% | 97.34% |
| Macro F1-score | 82.62% | 97.28% |
| Test Images | 2,038 | 2,038 |
| Categories | 18 | 18 |

### Key Findings

- MobileNetV2 V1 obtained higher scores in this exploratory evaluation.
- Its measured accuracy was 7.80 percentage points higher than the Basic CNN.
- The Basic CNN provided a useful from-scratch baseline.
- Transfer learning is a promising approach for this image classification task.

### Evaluation Limitations

This comparison is **exploratory, not a fully independent benchmark**.

The Basic CNN was trained using a newly generated data split. Some evaluation images may overlap with images previously used to train MobileNetV2 V1.

Additionally, Version 2's training set included images from the earlier Version 1 test split, so that original test set cannot be considered independent for Version 2.

A stronger future experiment would train and evaluate the models using a carefully controlled, leakage-free dataset split, with an untouched independent test set.

---

## 6. System Workflow

The FruitFresh AI prediction process follows these steps:

1. The user uploads a photograph or captures an image.
2. The image is converted into RGB format.
3. The image is resized to 224 × 224 pixels.
4. The image is processed by the selected trained model.
5. The model generates Softmax scores for the 18 categories.
6. The application identifies the highest-scoring category.
7. The system combines category scores to display the predicted fruit type and freshness condition.
8. The results appear on the Streamlit website.

**Important:** The displayed percentages are model-generated scores, not guarantees of prediction accuracy.

---

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | CNN development, training, and inference |
| MobileNetV2 | Pretrained CNN architecture |
| Google Colab | Development and model training |
| Google Drive | Storage of trained models |
| Kaggle | Image datasets |
| Streamlit | Web interface |
| NumPy | Array processing |
| Pandas | Evaluation reports and prediction history |
| Pillow | Image loading and processing |
| Scikit-learn | Dataset splitting and evaluation metrics |

---

## 8. Project Files

The main project files include:

```text
FruitFresh-AI/
|
|-- README.md
|-- app.py
|-- class_names.json
|-- requirements.txt
|
|-- notebooks/
|   |-- Fruit.ipynb
|   |-- FruitFresh_Final_Demo.ipynb
|
|-- results/
|   |-- basic_cnn_metrics.json
|   |-- basic_cnn_training_history.csv
|   |-- basic_cnn_classification_report.csv
|   |-- basic_cnn_confusion_matrix.npy
|
|-- models/
|   |-- fruitfresh_best.keras
|   |-- fruitfresh_v2_best.keras
|   |-- fruitfresh_basic_cnn.keras
```

*This is the recommended repository organization. Some files may need to be uploaded separately, and large trained models may remain in Google Drive rather than GitHub.*

---

## 9. How to Run the Website in Google Colab

The current application loads model files from Google Drive. The examples below assume your files are stored in:

`/content/drive/MyDrive/FruitFresh_Final_Project`

### Step 1 — Open Google Colab

Open your FruitFresh AI notebook in [Google Colab](https://colab.research.google.com/).

Connect to a runtime.

### Step 2 — Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

### Step 3 — Install Required Libraries

```python
!pip -q install streamlit tensorflow numpy pandas pillow scikit-learn
```

### Step 4 — Restore the Application

```python
import os
import shutil

SAVE_DIR = "/content/drive/MyDrive/FruitFresh_Final_Project"

required_files = [
    "fruitfresh_best.keras",
    "fruitfresh_v2_best.keras",
    "class_names.json",
    "app.py"
]

for filename in required_files:
    path = os.path.join(SAVE_DIR, filename)
    assert os.path.isfile(path), f"Missing: {filename}"
    print("READY:", filename)

shutil.copy2(
    os.path.join(SAVE_DIR, "app.py"),
    "/content/app.py"
)

print("FruitFresh AI restored successfully!")
```

### Step 5 — Start the Streamlit Server

```python
import subprocess
import time

server = subprocess.Popen(
    [
        "streamlit", "run", "/content/app.py",
        "--server.port=8501",
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ],
    stdout=open("/content/streamlit.log", "w"),
    stderr=subprocess.STDOUT
)

time.sleep(5)

print("Server running:", server.poll() is None)
```

The disabled CORS and XSRF protections are used only for the private Colab preview. They are not appropriate settings for public deployment.

### Step 6 — Open the Website in a New Tab

```python
from google.colab import output

output.serve_kernel_port_as_window(8501)
```

The FruitFresh AI website should open in a separate browser tab.

**Note:** The Google Colab runtime must remain connected while the website is in use.

---

## 10. How to Reproduce the Basic CNN Evaluation

The Basic CNN model is saved as:

`fruitfresh_basic_cnn.keras`

The experiment also saves:

- `cnn_train_manifest.csv`
- `cnn_val_manifest.csv`
- `cnn_test_manifest.csv`
- `basic_cnn_metrics.json`
- `basic_cnn_classification_report.csv`
- `basic_cnn_confusion_matrix.npy`
- `basic_cnn_training_history.csv`

The CSV manifests contain image paths from the original Colab dataset environment. These paths may need to be updated after restarting Colab or downloading the dataset again.

The training and evaluation cells are available in the development notebook.

For a reproducible experiment, the dataset version, split definitions, preprocessing, random seed, model configuration, and dependency versions should be preserved.

---

## 11. Limitations

FruitFresh AI currently has several limitations:

1. It supports only nine produce types.
2. It predicts one category for the entire image.
3. It cannot identify several fruits individually in a single photograph.
4. Similar decay patterns can cause classification mistakes.
5. Lighting, backgrounds, and image quality affect predictions.
6. An unsupported fruit may still be classified as one of the known categories.
7. Prediction scores are not guaranteed probabilities of correctness.
8. The current evaluation has possible data-overlap limitations.
9. Version 2 has not been independently evaluated across all 18 categories.
10. The system cannot detect internal spoilage or confirm food safety.

---

## 12. Future Improvements

Future development could include:

- A fully independent evaluation dataset
- Fair comparison using a shared leakage-free test split
- More varied real-world fruit and vegetable images
- Improved model generalization
- Unknown-category detection
- Multiple-fruit detection
- Permanent website deployment
- Persistent prediction history

---

## 13. Final Project Presentation

The complete presentation is available on Canva:

### [🍃 View FruitFresh AI Final Presentation](https://canva.link/bso3jl09ia1x3im)

The presentation covers:

- Problem statement and objectives
- Dataset and data preparation
- CNN architecture and transfer learning
- Basic CNN vs MobileNetV2 comparison
- Training strategy and fine-tuning
- Model evaluation and limitations
- Streamlit website
- Live demonstration
- Future improvements

---

## 14. Project Conclusion

FruitFresh AI demonstrates how Convolutional Neural Networks can classify fresh and rotten produce using image data.

The project includes a Basic CNN trained from scratch, a MobileNetV2 transfer learning model, and a fine-tuned MobileNetV2 model.

In the exploratory comparison, MobileNetV2 produced higher evaluation scores than the Basic CNN. However, more controlled independent testing is needed to determine how well these models generalize to unseen real-world images.

The project also includes a Streamlit application that allows users to test fruit images and compare the two MobileNetV2 models.

---

**FruitFresh AI | Deep Learning Final Project | Kirirom Institute of Technology**
