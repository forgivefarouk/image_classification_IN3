# Image Classification using Convolutional Neural Networks (CNNs)

## Overview

Welcome to the Image Classification project utilizing Convolutional Neural Networks (CNNs). This project aims to develop a robust image classification system capable of accurately categorizing images into predefined classes using state-of-the-art deep learning techniques.

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [CNN Architecture](#cnn-architecture)
- [Training and Validation](#training-and-validation)
- [Baseline Model](#baseline-model)
- [Baseline Model + BatchNormalization](#baseline-model--batchnormalization)
- [Baseline Model + BatchNormalization + Dropout](#baseline-model--batchnormalization--dropout)
- [Baseline Model + BatchNormalization + Dropout + Data Augmentation](#baseline-model--batchnormalization--dropout--data-augmentation)
- [Evaluation Metrics](#evaluation-metrics)

## Key Components

### 1. Dataset Selection and Preprocessing

The CIFAR-10 dataset is available through the [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html). You can download the dataset from the official website or use a library such as TensorFlow or PyTorch to access it programmatically.

#### Dataset Preprocessing

1. **Resizing:**
   - The images in the CIFAR-10 dataset are 32x32 pixels. Depending on the requirements of your CNN architecture, you may need to resize the images. Ensure consistency in image dimensions across the dataset.

2. **Normalization:**
   - Normalize the pixel values of the images to a range between 0 and 1. This ensures numerical stability during training and convergence.

3. **Augmentation (Optional):**
   - Consider applying data augmentation techniques to artificially increase the size of the training dataset. Common augmentations include random rotations, flips, and shifts, which can improve the model's generalization.

### 2. CNN Architecture Design

![CNN Architecture](reports\figures\model.png)

### 3. Training and Validation

- Split the dataset into training and validation sets (train =5K , validation=1K)
- Scale the input to range from 0 to 1

## Baseline Model

![Baseline Model](reports\figures\00-vanshing_grad.JPG)

- As observed in the figure, the model failed to learn anything after 10 epochs, and this occurred due to the issue of vanishing gradients

## Baseline Model + BatchNormalization

![Baseline Model + BatchNormalization](reports\figures\01-overfitting.JPG)

- Implementing normalization after the Conv2D layer to enhance the model's learning capacity
- If you observe in the model training you can see that Validation loss is increasing a lot which means our model is overfitting.

## Baseline Model + BatchNormalization + Dropout

- So let's regularize the model. In deep learning, dropout is a very good form of regularization.
![Baseline Model + BatchNormalization + Dropout](reports\figures\03-schedule_dropout.JPG)

## Baseline Model + BatchNormalization + Dropout + Data Augmentation

![Data Augmentation](reports\figures\05-augmented_data_100epochs.JPG)

- To reduce overfitting, we agreed that adding more data will reduce overfitting. Even when our model doesn't overfit, it's a very good idea to add more data if you can. But collecting data is tough, and also data preprocessing is not as fancy as it sounds. But what if we can generate data from our existing data? We already have 60K images, and we can generate lots and lots of images out of it.

### 5. Evaluation Metrics

## On Test Data
- accuracy: 0.8626 - precision: 0.8919 - recall: 0.8438 - F1_score: 0.8684

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository.
2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3. Run unittest:
    ```bash
    python src\unit_test.py
    ```
4. For training the model again:
    ```bash
    python src\model_building.py
    ```
5. For evaluating the model:
    ```bash
    python src\model_evaluation.py
    ```
6. For prediction:
    1. Assign the image you want to be classified to `img_path`.
    2. Run:
    ```bash
    python src\predict.py
    ```
