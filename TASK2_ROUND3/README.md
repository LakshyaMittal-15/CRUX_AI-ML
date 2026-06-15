# MNIST Continuous Digit Prediction Using CNN

## Overview

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to predict handwritten digits from the MNIST dataset.

Unlike the traditional MNIST classification approach (where the output is one of ten classes: 0–9), this project formulates the problem as a **regression task**, where the model predicts a continuous numerical value representing the digit.

The project also compares the CNN model performance against classical machine learning regression algorithms:

- Linear Regression
- K-Nearest Neighbours Regression (KNN)

The performance of all models is evaluated using regression metrics such as:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

# Project Objectives

The main objectives of this project are:

- Build a CNN model using PyTorch.
- Apply image preprocessing techniques to MNIST data.
- Train a deep learning model for continuous digit prediction.
- Evaluate CNN performance using regression metrics.
- Compare deep learning performance against classical machine learning methods.
- Understand the difference between classification and regression approaches on image datasets.

---

# Dataset

The project uses the **MNIST handwritten digit dataset**.

Dataset details:

| Dataset | Number of Images |
|---|---:|
| Training Data | 60,000 |
| Testing Data | 10,000 |

Each image contains:

- Grayscale pixels
- Image size: `28 × 28`
- Single channel

The target labels represent digits from:

```
0 - 9
```

---

# Technologies Used

## Programming Language

- Python

## Deep Learning Framework

- PyTorch

## Computer Vision

- TorchVision

## Machine Learning

- Scikit-learn

## Data Processing

- NumPy

## Visualisation

- Matplotlib

---

# Project Structure

Recommended repository structure:

```
MNIST-CNN-Regression/
│
├── README.md
│
├── requirements.txt
│
├── MNIST_CNN_Regression.ipynb
│
├── src/
│   ├── model.py
│   ├── train.py
│   └── evaluation.py
│
└── results/
    ├── training_loss.png
    └── predictions.txt
```

# Requirements

The project requires:

```
torch
torchvision
numpy
scikit-learn
matplotlib
```

---

# Data Preprocessing

The following preprocessing steps are applied:

## Tensor Conversion

The MNIST images are converted into PyTorch tensors.

## Normalisation

Pixel values are normalised using:

```
Mean = 0.1307

Standard deviation = 0.3081
```

Normalisation improves:

- Training stability
- Faster convergence
- Better model performance

## Data Loading

PyTorch DataLoader is used with:

```
Batch size = 64
```

Training data is shuffled to reduce bias.

---

# CNN Architecture

The CNN model consists of two main components:

1. Convolutional Feature Extraction
2. Fully Connected Regression Network

---

# Convolutional Layers

## First Convolution Block

```
Input:
1 × 28 × 28

Conv2D:
1 → 32 channels

Kernel:
3 × 3

Batch Normalization

Activation:
LeakyReLU

MaxPooling:
2 × 2
```

Output:

```
32 feature maps
14 × 14
```

---

## Second Convolution Block

```
Conv2D:
32 → 64 channels

Kernel:
3 × 3

Batch Normalization

Activation:
LeakyReLU

MaxPooling:
2 × 2
```

Output:

```
64 feature maps
7 × 7
```

---

# Fully Connected Layers

The extracted features are flattened:

```
7 × 7 × 64 = 3136 features
```

Network:

```
3136
 |
256 neurons
 |
64 neurons
 |
1 output neuron
```

The final output neuron predicts a continuous digit value.

---

# Model Training

## Loss Function

Since the problem is treated as regression:

```
Mean Squared Error Loss (MSELoss)
```

was used.

---

## Optimizer

The final CNN model uses:

```
Stochastic Gradient Descent (SGD)

Learning Rate:
0.001

Momentum:
0.9
```

---

## Training Configuration

```
Epochs:
10

Batch Size:
64
```

During training:

1. Forward propagation
2. Loss calculation
3. Backpropagation
4. Weight update

are performed for every batch.

---

# Model Evaluation

The trained model is evaluated on the MNIST test dataset.

The following metrics are calculated:

## Mean Squared Error (MSE)

Measures the average squared difference between predicted and true values.

Lower values indicate better performance.

---

## Mean Absolute Error (MAE)

Measures the average absolute prediction error.

---

## Root Mean Squared Error (RMSE)

Represents the standard deviation of prediction errors.


# CNN Model Results

Final CNN performance:

| Metric | Score |
|---|---:|
| MSE | 0.2156 |
| MAE | 0.2251 |
| RMSE | 0.4643 |


# Prediction Examples

Example outputs:

```
TRUE DIGIT => 7
PREDICTED VALUE => 7.1672


TRUE DIGIT => 2
PREDICTED VALUE => 1.9248


TRUE DIGIT => 9
PREDICTED VALUE => 8.1262
```

The model predicts continuous values close to the actual digit labels.

---

# Classical Machine Learning Baselines

To compare CNN performance, two traditional regression models were implemented.

---

# 1. Linear Regression

## Implementation

Using:

```
sklearn.linear_model.LinearRegression
```

MNIST images are flattened:

```
28 × 28 = 784 features
```

## Results

| Metric | Score |
|---|---:|
| MSE | 3.1508 |
| MAE | 1.4004 |
| RMSE | 1.7751 |

Linear Regression performs significantly worse compared with CNN because it cannot learn spatial image features.

---

# 2. K-Nearest Neighbours Regression

## Implementation

Using:

```
KNeighborsRegressor
```

Configuration:

```
Number of neighbours = 7

Weighting = distance
```

Before training:

```
Pixel values normalized by dividing by 255
```

KNN uses similarity between images to predict digit values.

---

# Model Comparison

| Model | MSE | Performance |
|---|---:|---|
| CNN | 0.2156 | Best |
| Linear Regression | 3.1508 | Poor |
| KNN Regression | - | Baseline |

The CNN achieves the best performance because convolutional layers automatically learn important visual features such as:

- Edges
- Shapes
- Stroke patterns

---

# Advantages of CNN Approach

Advantages:

- Learns image features automatically.
- Handles spatial relationships between pixels.
- Provides better prediction accuracy.
- More scalable for complex image tasks.

---

# Limitations

Current limitations:

- Problem is formulated as regression instead of classification.
- Only MNIST dataset is used.
- No hyperparameter optimisation was performed.
- Model architecture is relatively simple.

---

# Future Improvements

Possible improvements:

- Convert the task into a classification problem.
- Add more convolutional layers.
- Apply data augmentation.
- Use AdamW optimizer.
- Perform hyperparameter tuning.
- Add confusion matrix analysis.
- Deploy the model using Flask/FastAPI.

---



# Author

**LAKSHYA MITTAL**

GitHub:
https://github.com/LakshyaMittal-15





---

# License

This project is for educational purposes.
