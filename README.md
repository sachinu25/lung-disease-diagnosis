#  Xray Lung Classifier

## Problem statement
Pneumonia is an inflammatory condition of the lungs that primarily affects the alveoli (small air sacs). It is commonly caused by bacterial or viral infections and presents symptoms such as cough, chest pain, fever, and difficulty breathing. The severity varies depending on the patient’s health condition and immune response.

Early and accurate diagnosis is critical for effective treatment. Chest X-ray imaging is one of the most widely used diagnostic tools.
The objective of this project is to build an end-to-end deep learning–based API that classifies chest X-ray images as Pneumonia or Normal.

## Solution Proposed
This project uses Computer Vision and Deep Learning to automatically detect pneumonia from chest X-ray images.

Key Highlights:

A custom Convolutional Neural Network (CNN) is built using PyTorch

The trained model is exposed via a FastAPI service

The application is Dockerized for portability

Deployed on AWS Cloud using modern MLOps practices

CI/CD enabled using GitHub Actions

🧠 Model Architecture

Custom CNN architecture

Optimized for binary image classification

Designed to balance accuracy and inference speed

📁 xray_arch/ contains the CNN architecture definition.

![xray_arch](https://user-images.githubusercontent.com/71321529/216753362-aeb34400-d21d-4b21-b2ce-63b86a47b594.jpg)

## Dataset used
Dataset provided by Apollo Diagnostic Center for research purposes

Contains labeled Chest X-ray images

Used strictly for Proof of Concept (POC)

⚠️ Dataset is not publicly shared due to confidentiality

## Tech Stack Used
Programming Language: Python

Deep Learning Framework: PyTorch

API Framework: FastAPI

Containerization: Docker

Cloud Platforms: AWS, Azure

CI/CD: GitHub Actions

## Infrastructure required
AWS S3 – Model and artifact storage

AWS App Runner – Application deployment

GitHub Actions – Continuous Integration & Deployment


📂 Project Structure

xray/
│
├── components/
│   ├── data_ingestion
│   ├── data_transformation
│   ├── model_training
│   ├── model_evaluation
│   └── model_pusher
│
├── logger/
├── exception/
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md

## How to run

Step 1. Download the zip file
```
Download the zip file and extract it to a folder.
```
Step 2. Create a conda environment.
```
conda create -p env python=3.8 -y
```
```
conda activate ./env
````
Step 3. Install the requirements 
```
pip install -r requirements.txt
```
Step 4. Export the environment variable
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>

```
Step 5. Run the application server
```
python app.py
```
Step 6. Train application
```bash
http://localhost:8001/train
```
Step 7. Prediction application
```bash
http://localhost:8001/predict
```
## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image
```
docker build -t xray_classification .
```

3. Run the Docker image
```
docker run -d -p 8001:8001 -e AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> -e AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> xray_classifier
```

## Models Used
* Custom CNN architecture

## `xray` is the main package folder which contains 


**Components** : Contains all components of Deep Learning(CV) Project
- data_ingestion
- data_transformation
- model_training
- model_evaluation
- model_pusher


**Custom Logger and Exceptions** are used in the Project for better debugging purposes.

## Conclusion
This project demonstrates a real-world medical AI application capable of assisting doctors in pneumonia diagnosis using chest X-ray images. While not intended to replace clinical judgment, it can serve as a decision-support tool to improve diagnostic efficiency and accuracy.
