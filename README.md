# 🩻 Bone Fracture Detection & Classification using YOLOv8

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![Computer Vision](https://img.shields.io/badge/Field-Computer%20Vision-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-96.34%25-brightgreen)

## 📌 Project Overview
This project utilizes the **YOLOv8 (You Only Look Once)** deep learning architecture to detect and classify bone fractures in X-ray images. The system is designed to assist medical professionals by providing a "second opinion," automatically localizing the fracture with a bounding box and identifying the specific type of fracture with high confidence.

## 🎯 Key Features
* **Object Detection:** Accurate localization of fracture regions within X-ray images.
* **Multi-Class Classification:** Classifies fractures into specific types (e.g., [List your types here: Transverse, Oblique, Comminuted, etc.]).
* **Visual Output:** Generates output images with bounding boxes, class labels, and confidence scores overlaid.
* **High Performance:** Achieved an accuracy of **96.34%** on the test dataset.

## 🛠️ Tech Stack
* **Language:** Python
* **Core Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **Libraries:** OpenCV, Pandas, Matplotlib, NumPy
* **Environment:** [e.g., Google Colab / Jupyter Notebook / VS Code]

## 📊 Dataset
The model was trained on a dataset containing X-ray images of fractured and healthy bones.
* **Preprocessing:** Images were resized and normalized. Data augmentation (flipping, rotation, brightness adjustment) was applied to improve model robustness.
* *https://www.kaggle.com/datasets/orvile/human-bone-fractures-image-dataset-hbfmid*

## 🚀 Model Performance
The YOLOv8 model was trained for `50` epochs.
* **mAP (mean Average Precision):** 96.34%
* **Inference Speed:** Real-time detection capabilities.

## 📁 Directory Structure
->.venv
->app.py
->requirements.txt
->best.pt(model downloaded)
