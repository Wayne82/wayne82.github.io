---
layout: post
title:  "Neural Network Notes: Backpropagation"
categories: neural-network
mathjax: true
---

As an individual, I feel incredibly lucky to live in an era largely free from major national or global wars, while both fundamental sciences and engineering are advancing rapidly, reaching new peaks one after another at an unprecedented pace. Among these breakthroughs, Artificial Intelligence — particularly Large Language Models (LLMs) — stands at the forefront, sparking discussions about AI, AGI, and the future of intelligence. This technology is both fascinating and revolutionary — not only because of the groundbreaking moment when LLMs first enabled fluent human-machine conversations, making what once seemed like science fiction a reality, but also due to their rapidly expanding capabilities, which have the potential to transform nearly every industry.

Out of deep curiosity, I set out boldly to understand the underlying techniques that led to these breakthroughs and to explore whether any fundamental principles of intelligence have been uncovered. But, the first step for me is to review the basics of neural networks and grasp one of the core algorithms — backpropagation. This blog serves primarily as a record of my own learning notes, rather than a comprehensive explanation on this subject.

# Neural Network Basics
Briefly, a neural network is a machine learning model inspired by the structure and functioning of human brain, composed of interconnected nodes (neurons) that process sample data through weighted connections in layers, allowing it to learn and make predictions on new data. It is widely used for tasks such as image recognition, natural language processing, and so on.
## Key Concepts
As what I have learned so far, the key concepts come to me include,
* Layered structure, a typeical neural network consists of the following layers:
  * **Input Layer**: a list of neurons that can encode the information of sample data. E.g., if the sample data is an image, the neurons in the input layer will encode the values of the image pixels.
  * **Hidden Layers**: these are the middle layers, could be one or many, and the neurons in these layers are neither inputs nor outputs. Mathmatically, they perform computations and transformations through weighted connections. And the more layers, the deeper the network (hence, "deep learning")
  * **Output Layer**: produces the final results. E.g. a classification label.
* **Connections**, each neuron in a layer is connected to neurons in the next layer through weights and biases, which can be adjusted during training process.
* **Activation Function**, is a function applying on each neuron in the hidden layers to introduce non-linearity, allowing the network to learn complex patterns. Common activition functions include,
  * **Sigmoid**, historically used in binary classification problems, because it maps any input value to a range of $[0, 1]$. $$\sigma(x)=\frac{1}{1 + e^{-x}}$$
  * **ReLU (Rectified Linear Unit)**, default choice for hidden layers in deep neural networks, can help learning fast. $$\text{ReLU}(x) = \max(0, x)$$
  * **Softmax**, used exclusively in the output layer for multi-class classification problems. $$\text{softmax}(\mathbf{z}) = \left( \frac{e^{z_1}}{\sum_{j=1}^n e^{z_j}}, \frac{e^{z_2}}{\sum_{j=1}^n e^{z_j}}, \dots, \frac{e^{z_n}}{\sum_{j=1}^n e^{z_j}} \right)$$ where $\mathbf{z}=(z_1, z_2, \dots, z_n)$
* **Feedforward**, is the process by which data flows through a neural network in one direction: from the input layer, through the hidden layers, and to the output layer. This is the foundational structure of most neural networks, where data moves forward without any loops. Mathematically, the feedforward is doing calculations at each neuron using activation function with weighted input.
  * **Weighted Input**, $z=\mathbf{w} \cdot \mathbf{x} + b$ where $z$ is the activation of a neuron, $\mathbf{w}$ is the vector of weights connecting to that neuron from previous layer, and $b$ is the bias.
* **Cost Function**, is also known as **loss function** is a mathematical function that measures how well a neural network performs on a training dataset. It quantifies the difference between the outputs of the network and the actual desired outputs for the training dataset. And the goal of training a neural network is to **minimize the cost function**. One of the common cost functions is **Quadratic Loss Function**, also known as **Mean Squared Error (MSE)** or **L2 Loss**. $$L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
* **Gradient Descent**
  * Stochastic gradient descent
* **Backpropagation**
* **Training**, epoch, batch, iteration.

## The Architecture of Neural Network
## Learning with Gradient Descent

# The Backpropagation Algorithm
## Derivative Chain Rule
## The 4 Fundamental Equations Behind Backpropagation
## Proof of the 4 Fundamental Equations
## The Vanishing Gradient Problems

# Learning Materials

# Final Words
