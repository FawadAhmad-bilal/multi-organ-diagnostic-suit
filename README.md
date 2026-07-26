<div align="center">

# 🩺 Multi-Organ Diagnostic Suite

**Five deep learning models. One dashboard. Grad-CAM explainability built in.**

A Streamlit + TensorFlow/Keras web app that classifies medical images across
five different diagnostic tasks using transfer learning, and visualizes
*why* each model made its decision with Grad-CAM heatmaps.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

</div>

---

🔗 **[Live Demo](https://multi-organ-diagnostic-suit.streamlit.app)**

## 📸 Preview

> _Add a screenshot or screen recording of the dashboard here once deployed
> — recruiters and reviewers open this before anything else._
>
> `![App preview](docs/preview.png)`

---

## 🧠 What it does

Upload a medical image, pick the matching module in the sidebar, and the
app will:

1. Preprocess the image exactly as it was preprocessed during training
   (correct resize + backbone-specific normalization).
2. Run it through the corresponding trained CNN.
3. Show the predicted class with a confidence score and a full
   per-class probability breakdown.
4. Overlay a **Grad-CAM heatmap** next to the original image, so you can
   see which region of the image the model actually focused on.

## 🗂️ Modules

| Module | Backbone | Input Size | Classes |
|---|---|---|---|
| 🫁 Chest X-ray (COVID-19 Radiography) | ResNet50 | 224×224 | COVID, Lung Opacity, Normal, Viral Pneumonia |
| 🦟 Malaria Cell Image | MobileNetV2 | 128×128 | Parasitized, Uninfected |
| 🩹 Skin Lesion (HAM10000) | EfficientNet | 128×128 | akiec, bcc, bkl, df, mel, nv, vasc |
| 🧠 Brain Tumor MRI | VGG16 | 128×128 | glioma, meningioma, notumor, pituitary |
| 👁️ Eye Fundus (Diabetic Retinopathy) | ResNet50 | 224×224 | No DR, Mild, Moderate, Severe, Proliferative DR |

Each model was trained separately via transfer learning on a public Kaggle
dataset for its respective task, then wired into a single unified
inference + explainability interface.

## ✨ Features

- Custom dark-themed UI (no default Streamlit look)
- Sidebar module switcher with per-organ icons
- Styled prediction card with confidence score
- Per-class probability bars
- Side-by-side original image vs. Grad-CAM heatmap
- Backbone-agnostic Grad-CAM implementation — one function serves all
  four architectures used across the five models

## 🛠️ Tech Stack

- **Modeling:** TensorFlow / Keras, transfer learning (ResNet50,
  MobileNetV2, EfficientNet, VGG16)
- **Explainability:** Grad-CAM (custom implementation, see
  [`grad_cam.py`](grad_cam.py))
- **Interface:** Streamlit with custom CSS
- **Data:** Public Kaggle datasets, downloaded via the Kaggle API

## 🚀 Getting Started

```bash
git clone https://github.com/FawadAhmad-bilal/multi-organ-diagnostic-suite.git
cd multi-organ-diagnostic-suite
pip install -r requirements.txt
streamlit run app.py
```

Place all five `.keras` model files in the project root (see
[Project Structure](#-project-structure)) — they are not committed to this
repo due to size; see [Model Files](#-model-files) below.

## 📁 Project Structure

```
multi-organ-project/
├── app.py              # Streamlit UI + model config/loading
├── grad_cam.py          # Backbone-agnostic Grad-CAM implementation
├── requirements.txt
├── CONCEPTS.md           # Deep-dive notes on how Grad-CAM works internally
├── chest_model.keras
├── malaria_model.keras
├── skin_cancer.keras
├── vgg_model.keras
└── eye_prediction.keras
```

## 📦 Model Files

The trained `.keras` files are not included in this repository because of
their size. To run the app, either train the models yourself using the
matching Kaggle datasets below, or host the weights externally (e.g. a
release asset or a cloud bucket) and download them into the project root
before launching.

| Model | Dataset |
|---|---|
| Chest X-ray | COVID-19 Radiography Database (Kaggle) |
| Malaria | Malaria Cell Images (Kaggle) |
| Skin Lesion | HAM10000 (Kaggle) |
| Brain Tumor | Brain Tumor MRI Dataset (Kaggle) |
| Eye Fundus | APTOS 2019 Blindness Detection (Kaggle) |

## 📖 Learn More

See [`CONCEPTS.md`](CONCEPTS.md) for a written breakdown of how the
Grad-CAM implementation works internally — the gradient math, why the last
conv layer is used, and why preprocessing consistency matters — useful if
you want to understand or extend it rather than just run it.

## ⚠️ Disclaimer

This is a student portfolio project built to demonstrate transfer learning
and model explainability. It is **not** a certified medical device and
must not be used for real diagnostic decisions.

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Fawad Ahmad**
BS Artificial Intelligence, University of Haripur
[GitHub](https://github.com/FawadAhmad-bilal)
