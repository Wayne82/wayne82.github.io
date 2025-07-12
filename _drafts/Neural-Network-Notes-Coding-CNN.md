---
layout: post
title:  "Neural Network Notes: Coding CNN"
categories: neural-network
mathjax: true
comments: true
---

Last time, I've summarized my learnings of the [basic concepts of Convolutional Neural Networks (CNNs)](https://wayne82.github.io/neural-network/2025/05/15/Neural-Network-Notes-Convolutional-Neural-Network.html) and a [basic neural network implementation from scratch](https://wayne82.github.io/neural-network/2025/05/15/Neural-Network-Notes-Basic-Neural-Network.html). In this post, I will continue sharing my experience and notes of coding a CNN from "scratch", which is still part of my [nn-learn](https://github.com/waynewang/nn-learn) project.

## The Structure of the Code
The basic code structure of the CNN is very similar to the basic neural network implementation, but with dedicated network layers and functionalities specific to CNNs. See the diagram below for an overview of the code structure.
![image](/assets/images/CNN%20code%20structure.png)

The main components include:
- **Layer Classes**: Each layer type (e.g., convolutional, pooling, fully connected, etc. details see below section) is implemented as a separate class, encapsulating its forward and backward pass logic, an optional update function to adjust weights and biases and a reset gradients function.
- **Convnet Class**: The `ConvNet` class manages the overall architecture, including adding layers and providing the SGD training process.
- **Training Process**: The training process is handled in a loop of total N epochs, each epoch runs the whole training dataset, which split into batches, and each batch of data is passed through each layer of the network in the forward pass, gradients for each layer are computed in the backward pass, and at the end of each batch processing the weights and biases are updated based on the computed gradients.

## The Key Layers of the CNN Implementation
Convolutional Neural Networks are commonly made up of only three layer types, **CONV**, **POOL** (assume max pooling), and **FC** (fully connected). However, in practice, CNNs often explicitly write **ReLU** activation function as a separate layer, which applies element-wise non-linearity, and a **Flatten** layer to convert the multi-dimensional feature maps into a one-dimensional vector before passing it to the fully connected layer.

Then, the key layers of the CNN implementation in this exercise can be summarized as below:

1. **Convolutional Layers**: These layers apply convolution operations to the input, allowing the network to learn spatial hierarchies of features. Each convolutional layer is defined by its number of filters, and kernel size.

2. **Pooling Layers**: Pooling layers reduce the spatial dimensions of the feature maps, helping to down-sample the representation and reduce the number of parameters. Max pooling is commonly used, which takes the maximum value in each patch of the feature map.

3. **Activation Functions**: Non-linear activation functions, such as ReLU (Rectified Linear Unit), are applied after each convolutional layer to introduce non-linearity into the model.

4. **Flatten**: After the convolutional and pooling layers, the multi-dimensional feature maps are flattened into a one-dimensional vector to be fed into fully connected layers.

5. **Fully Connected Layers**: After several convolutional and pooling layers, the high-level reasoning in the neural network is done via fully connected layers. These layers connect every neuron in the previous (flattened) layer to every neuron in the final output layer.

6. **Softmax and Loss Function**: The loss function measures the difference between the predicted output and the true output. One of the loss functions for classification tasks and also used in this implementation is the **Cross-Entropy loss**. And before applying the loss function, the **Softmax** function is often applied to the output layer to convert the raw logits into probabilities.

## Backpropagation in CNNs
Now, we come to the most important and interesting part of the CNN implementation, which is the backpropagation algorithm. Essentially, the 4 fundamental backpropagation equations from Michael Nielsen’s book are still applicable, but with some modifications to accommodate the convolutional and pooling layers. The key idea of backpropagation in CNNs is still the same,
* First, compute the gradients of the loss function with respect to the output (the logits) of the network.
* Then, propagate these gradients backward through the network, layer by layer. Specifically, there are **at most 3** different gradients to calculate for each layer:
  - The gradient of the loss with respect to the **weights** of the current layer.
  - The gradient of the loss with respect to the **biases** of the current layer.
  - The gradient of the loss with respect to the **input** of the current layer, which will be used as the input for the previous layer in the backward pass. This is the gradient that makes the backpropagation happen through the network.
* For each layer, the gradients are computed based on the specific operations performed in that layer.
  - For **convolutional layer**, the gradients w.r.t. weights are computed by summing the convolution of the input feature maps of current layer with the gradient of input feature maps from the next layer; the gradients w.r.t. biases are computed by summing the gradients of the input feature maps from the next layer; and the gradients w.r.t. input are computed by summing the convolution of the gradient of input feature maps from the next layer with the weights of the current layer.
  - For **ReLU activation layer**, the gradients are computed by applying the ReLU derivative (which is 1 for positive inputs and 0 for negative inputs) to the gradient of the input feature maps from the next layer.
  - For **Flatten layer**, the gradients are simply reshaped to match the shape of the input feature maps.
  - For **fully connected layers**, the gradients w.r.t. weights are computed by multiplying the input feature maps of current layer with the gradient of input feature maps from the next layer; the gradients w.r.t. biases are equal to the gradients of the input feature maps from the next layer; and the gradients w.r.t. input are computed by multiplying the gradient of input feature maps from the next layer with the transpose of the weights of the current layer.
  - For **pooling layer**, the gradients are propagated back to the input based on the indices of the maximum values in the pooling operation.
* Finally, the gradients of the weights and biases are used to update the parameters of the network.

> 📝 Notes
>
> The gradients of the weights and biases need to reset to zero before each batch processing, otherwise the gradients will accumulate across batches, which is not what we want. This is done in the `zero_grad` function for layers having trainable parameters (e.g., convolutional and fully connected layers).

## Training and Optimizations

Optimization path - simple loops -> batch processing -> vectorization

## Test Results

## A bit of Notes about Numpy N-D Arrays

## What is Next?