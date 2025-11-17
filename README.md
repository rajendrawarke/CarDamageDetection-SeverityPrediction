# Car Damage Detection & Severity Prediction (Motor Insurance)

## 📌 Project Overview
This project applies **computer vision** to detect car damage and classify severity levels using the **Roboflow / Skillfactory Car Damage dataset**. The goal is to build an AI model that automates the initial assessment of motor insurance claims.

The project is developed as part of the **AAI-521 Final Project**, and completed **individually**, following all requirements for modeling, EDA, pre-processing, and presentation deliverables.

---

## 🎯 Objectives
- Detect visible car damage using deep learning.
- Classify severity into categories (e.g., **minor**, **moderate**, **severe**).
- Prepare a reproducible workflow using **Google Colab**.
- Document the process with a full **technical report** and **video presentation**.

---

## 📂 Dataset
**Dataset Name:** Roboflow / Skillfactory Car Damage  
**Source:** Roboflow Universe  
**Link:** https://universe.roboflow.com/skillfactory-pro-v/carg-damage

### Dataset Details
- Contains **image samples of vehicle exteriors** with labeled damage regions.
- Annotations provided in **YOLO format**.
- Includes multiple classes:
  - `scratch`
  - `dent`
  - `glass_shatter`
- Supports both **detection** and **damage severity classification**.

---

## 🛠 Tech Stack
- **Python**
- **Google Colab**
- **TensorFlow / Keras** or **PyTorch**
- **OpenCV**
- **Roboflow API** (dataset import)
- **Matplotlib / Seaborn** (visualization)

---

## 🧹 Data Preprocessing
- Import dataset via Roboflow API.
- Resize images (e.g., 416x416 or 640x640 for YOLO).
- Normalization & augmentation:
  - rotation
  - brightness adjust
  - flip
  - zoom
- Convert annotations to required format.

---

## 🤖 Modeling Approach
### Step 1: Damage Detection (Object Detection)
Options:
- **YOLOv8 / YOLOv10** (preferred)
- TensorFlow object detection models

### Step 2: Severity Prediction (Classification)
Approaches:
- Train a **small CNN classifier** using cropped damage regions.
- Or attach a classification head on detection model outputs.

### Training
- Train/Validation/Test Split
- Use early stopping & model checkpoints

### Evaluation Metrics
- **mAP@50** / **mAP@50-95** for detection
- **Accuracy**, **Precision**, **Recall**, **F1-score** for severity classification

---

## 📊 Results & Visualizations
The README will be updated with:
- Sample predictions
- Confusion matrix
- mAP curves
- Loss/accuracy plots

---

## 📁 Project Structure
```
Car-Damage-Detection/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Training_Model.ipynb
│   └── Evaluation.ipynb
│
├── models/
│   └── best_model.pt / .h5
│
├── data/  (ignored in Git)
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   └── inference.py
│
├── README.md
└── requirements.txt
```

---

## 🎥 Final Deliverables
As required by the course:
- **Final Technical Report (PDF)**
- **GitHub Repository**
- **10–12 min Presentation Video**
- **Appendix with code & visualizations**

---

## 🙋 Author
This project is completed **independently** by:  
**Rajendra Warke**

For academic requirements under **AAI-521 (MS in Applied AI)**.
