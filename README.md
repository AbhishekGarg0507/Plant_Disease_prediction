# 🌿 Plant Disease Detection System

<div align="center">

![Plant Disease Detection](https://img.shields.io/badge/AI-Plant%20Disease%20Detection-2e7d32?style=for-the-badge&logo=leaf&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-success?style=for-the-badge)

**An AI-powered web application that detects plant diseases from leaf images using a custom-trained Convolutional Neural Network.**

[🚀 Live Demo](https://plantdiseaseprediction-2.streamlit.app/) · [📊 Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) · [📓 Training Notebook](#)

</div>

---

## 📸 Demo

<div align="center">

| Upload a Leaf | Get Instant Results |
|:---:|:---:|
| Upload or capture a photo of a plant leaf | Get disease name, confidence score & treatment advice |

</div>

---

## ✨ Features

- 🔍 **38 Disease Classes** across 14 plant species detected with ~95% accuracy
- 📷 **Camera & Upload Support** — take a photo directly or upload from your device
- 📊 **Confidence Meter** — color-coded confidence score with high/medium/low indicators
- 🏆 **Top 3 Predictions** — see the most likely diagnoses with probability scores
- 💊 **Treatment Advice** — cause, symptoms, treatment, and prevention for every disease
- 📈 **Training History** — interactive accuracy & loss charts on the About page
- 🚫 **Invalid Image Detection** — rejects non-leaf images below confidence threshold

---

## 🧠 Model Architecture

A custom CNN built from scratch using TensorFlow/Keras:

```
Input (128×128×3)
    ↓
Conv Block 1: Conv2D(32) → Conv2D(32) → MaxPool
    ↓
Conv Block 2: Conv2D(64) → Conv2D(64) → MaxPool
    ↓
Conv Block 3: Conv2D(128) → Conv2D(128) → MaxPool
    ↓
Conv Block 4: Conv2D(256) → Conv2D(256) → MaxPool
    ↓
Conv Block 5: Conv2D(512) → Conv2D(512) → MaxPool
    ↓
Dropout(0.25) → Flatten → Dense(1500) → Dropout(0.4)
    ↓
Output: Dense(38, softmax)
```

| Metric | Score |
|--------|-------|
| Training Accuracy | **96.5%** |
| Validation Accuracy | **95.0%** |
| Training Loss | 0.104 |
| Validation Loss | 0.161 |
| Epochs | 6 |
| Optimizer | Adam (lr=0.0001) |

### 📈 Training History

![Training History](assets/training_history.png)

---

## 🌱 Supported Plants & Diseases

<details>
<summary>Click to expand all 38 classes</summary>

| Plant | Diseases |
|-------|----------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌽 Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍊 Orange | Huanglongbing (Citrus Greening) |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🫐 Raspberry | Healthy |
| 🫘 Soybean | Healthy |
| 🎃 Squash | Powdery Mildew |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

</details>

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/abhishekgarg0507/plant_disease_prediction.git
cd plant_disease_prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run main.py
```

---

## 📁 Project Structure

```
plant_disease_prediction/
├── main.py                  # Streamlit app
├── disease_info.json        # Disease details (cause, treatment, prevention)
├── training_history.json    # Model training metrics
├── trained_model.keras      # Trained CNN model
├── logo.png                 # App logo
├── home_image.jpg           # Home page image
└── requirements.txt         # Dependencies
```

---

## 📦 Tech Stack

| Technology | Purpose |
|------------|---------|
| **TensorFlow / Keras** | Model training & inference |
| **Streamlit** | Web application framework |
| **Plotly** | Interactive training history charts |
| **NumPy** | Image preprocessing & array operations |
| **Python** | Core language |

---

## 📊 Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Size:** ~87,000 RGB images
- **Classes:** 38 (healthy + diseased)
- **Split:** 80% train / 20% validation

---

## 🔮 Future Scope

- 🔄 Transfer learning with EfficientNet/MobileNetV2 for higher accuracy
- 📱 Mobile app integration for real-time field use
- 🌍 Extended species and disease coverage
- 🌐 Multi-language support for global farmers

---

## 👨‍💻 Author

**Abhishek Garg**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abhishekgarg0507)

---

<div align="center">

⭐ **If you found this project useful, please give it a star!** ⭐

</div>
