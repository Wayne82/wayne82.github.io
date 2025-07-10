---
layout: post
title:  "Neural Network Notes: Coding CNN"
categories: neural-network
mathjax: true
comments: true
---

Last time, I've summarized my learnings of the [basic concepts of Convolutional Neural Networks (CNNs)](https://wayne82.github.io/neural-network/2025/05/15/Neural-Network-Notes-Convolutional-Neural-Network.html) and a [basic neural network implementation from scratch](https://wayne82.github.io/neural-network/2025/05/15/Neural-Network-Notes-Basic-Neural-Network.html). In this post, I will continue sharing my experience of coding a CNN from scratch, which is still part of my [nn-learn](https://github.com/waynewang/nn-learn) project.

## The Structure of the Code

## Key Components of the CNN Implementation

1. **Convolutional Layers**: These layers apply convolution operations to the input, allowing the network to learn spatial hierarchies of features. Each convolutional layer is defined by its number of filters, kernel size, and activation function.

2. **Pooling Layers**: Pooling layers reduce the spatial dimensions of the feature maps, helping to down-sample the representation and reduce the number of parameters. Max pooling is commonly used, which takes the maximum value in each patch of the feature map.

3. **Activation Functions**: Non-linear activation functions, such as ReLU (Rectified Linear Unit), are applied after each convolutional and fully connected layer to introduce non-linearity into the model.

4. **Flattening**: After the convolutional and pooling layers, the multi-dimensional feature maps are flattened into a one-dimensional vector to be fed into fully connected layers.

5. **Fully Connected Layers**: After several convolutional and pooling layers, the high-level reasoning in the neural network is done via fully connected layers. These layers connect every neuron in one layer to every neuron in the next layer.

6. **Softmax and Loss Function**: The loss function measures the difference between the predicted output and the true output. Common loss functions for classification tasks include cross-entropy loss. The softmax function is often applied to the output layer to convert the raw logits into probabilities.

## Backpropagation in CNNs

## Training and Optimizations

## Test Results

## What is Next?