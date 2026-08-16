# Deep Learning Face Recognition System

A real-time face recognition system developed using Python and deep learning.

## Project Overview

This project uses a webcam to detect and recognize faces in real time. FaceNet is used to generate face embeddings, while MTCNN is used for face detection. Cosine similarity is used to compare the live face with stored face embeddings.

## Technologies Used

- Python 3.11
- OpenCV
- TensorFlow
- FaceNet
- MTCNN
- NumPy
- Scikit-learn
- Cosine Similarity

## Features

- Real-time face detection
- Face registration using webcam
- Face image collection
- FaceNet-based feature extraction
- Face embedding generation
- Real-time face recognition
- Unknown face detection
- Similarity score display

## System Workflow

Webcam  
↓  
MTCNN Face Detection  
↓  
FaceNet Feature Extraction  
↓  
Face Embedding  
↓  
Cosine Similarity  
↓  
Recognized Name / Unknown

## Project Structure

```text
FaceRecognition/
│
├── camera_test.py
├── encode_faces.py
├── face_detect.py
├── recognize.py
├── register.py
├── requirements.txt
├── README.md
└── .gitignore