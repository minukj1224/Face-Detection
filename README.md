# Face Landmark Detection

## 📌 Project Overview
This project focuses on **face landmark detection** using deep learning.  
It implements and compares **ResNet, Hourglass Network, and Vision Transformer (ViT)** for detecting facial landmarks.

## 📁 Dataset Structure
※ Deep Convolutional Network Cascade for Facial Point Detection. The data was refined using the dataset released in the paper and used for learning.
- **Training Data:** `dataset/train/`
- **Test Data:** `dataset/test/`
- **Annotation Files:** `train.txt`, `test.txt`
  - Format: `image_path landmark1_x landmark1_y ... landmark5_x landmark5_y`
  - **Bounding boxes are removed**, keeping only **5 facial landmarks**
  - **Landmarks are normalized to a `96x96` resolution**