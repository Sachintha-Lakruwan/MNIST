# MNIST Neural Network from Scratch

A simple implementation of a neural network for digit recognition using the MNIST dataset, built entirely from scratch using only NumPy (no TensorFlow or Keras).

## Overview

This project demonstrates how to build a neural network from the ground up to classify handwritten digits (0-9) from the MNIST dataset. The implementation includes:

- **Two-layer neural network architecture**
- **Forward and backward propagation**
- **Gradient descent optimization**
- **ReLU and Softmax activation functions**

## Architecture

The neural network has a simple two-layer architecture:

- **Input layer**: 784 units (28×28 pixel images)
- **Hidden layer**: 10 units with ReLU activation
- **Output layer**: 10 units with Softmax activation (one for each digit 0-9)

## Features

- ✅ Built entirely with NumPy (no deep learning frameworks)
- ✅ Implements forward and backward propagation from scratch
- ✅ Uses ReLU activation for hidden layer
- ✅ Uses Softmax activation for output layer
- ✅ Gradient descent optimization
- ✅ Achieves ~89% accuracy on test set
- ✅ Visualizes predictions with matplotlib

## Requirements

```
numpy
pandas
matplotlib
```

## Key Functions

- `init_params()`: Initialize network weights and biases
- `forward_prop()`: Forward propagation through the network
- `backward_prop()`: Backward propagation to compute gradients
- `gradient_descent()`: Main training loop
- `make_predictions()`: Make predictions on new data
- `test_prediction()`: Visualize predictions on sample images

## Results

The trained model achieves approximately **89% accuracy** on the test set, demonstrating good generalization from the training data.

## Learning Objectives

This project serves as an educational tool to understand:

- Neural network fundamentals
- Forward and backward propagation
- Activation functions (ReLU, Softmax)
- Gradient descent optimization
- Matrix operations in neural networks

## Dataset

Uses the MNIST digit recognition dataset with:

- Training set: ~40,000 samples
- Development set: 2,000 samples
- Each image: 28×28 grayscale pixels
