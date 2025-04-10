# Convolutional_NN
# Fashion MNIST Classifier with PyTorch

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying images from the Fashion MNIST dataset.

## Overview

The project implements a CNN to classify fashion items from the Fashion MNIST dataset, which consists of 28x28 grayscale images across 10 clothing categories. The code includes data preprocessing, model architecture, training, and evaluation components.

## Features

- Data loading and preprocessing using Pandas and PyTorch
- Visualization of sample images from the dataset
- Custom CNN architecture with:
  - Two convolutional layers with ReLU activation and batch normalization
  - Max pooling layers
  - Fully connected layers with dropout
- Training with SGD optimizer and CrossEntropyLoss
- Evaluation on test set

## Requirements

- Python 3.x
- PyTorch
- Pandas
- Scikit-learn
- Matplotlib

## Usage

1. Ensure the Fashion MNIST dataset (`fashion-mnist_train.csv`) is in the working directory
2. Install required dependencies:
```bash
pip install torch pandas scikit-learn matplotlib optuna
