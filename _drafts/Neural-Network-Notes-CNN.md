---
layout: post
title:  "Neural Network Notes: CNN"
categories: neural-network
mathjax: true
comments: true
---

After completing my initial exploration and study of neural networks — and documenting it in [Neural Network Notes: the Basics and Backpropagation](https://wayne82.github.io/neural-network/2025/03/30/Neural-Network-Notes-The-Basics-and-Backpropagation.html) — I am now moving on to the next fascinating topic: Convolutional Neural Networks (CNNs). CNNs are a type of deep learning architecture specifically designed to process data with a grid-like structure, such as images. They excel at tasks like image recognition and classification, and have remained a cornerstone of computer vision and deep learning to this day.

Thanks to the open internet, I was able to access a world-class course — [CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/) — from a top university. After spending about two weeks studying eagerly during my evenings, I completed the course. As usual, I would like to document my notes and learning experiences in this blog.

# Prior Arts
The course has 2 modules, the first one takes up quite a considerable amount of space to introduce the prior arts of image classification, including k-Nearest Neighbor, and linear classification using SVM loss and softmax loss function, further draw forth the vallina neural network introduction. Then, the second module goes to the main theme about CNN.

First, the image classification problem all these different approaches are developed to solve is the task of assigning an input image one label from a fixed set of categories. Despite it is a simple task for a human, it is extremely difficult for a computer to achieve with a reasonable successful rate. Traditionally, we would think how might we go about writing an algorithm that can classify images into distinct categories? However, at least right now it is not obvious what such an algorithm looks like. Thus, the practical approach that is taken is a data-driven approach, which relies on a training dataset that is well labeled as different classes, let the learning model to learn every class, and then predict the class for new input data.

It is interesting to see how different classification approaches have their distinct pros and cons repsectively. The KNN classifier takes no time to train, since all that is required is to store and index the training data. However, the cost is paied at prediction time which it needs to compare the input data against every single training exmaple. Just on the other extreme, regarding other classifiers, like Linear Classification, Neural Network, and especially the Deep Neural Network, they are very expensive to train, but once the training is finished, it will be fast and cheap to classify a new input data.

Unlike KNN, the other classifiers will commonly have 2 major components: a score function that maps the raw data to class scores, and a loss function that quantifies the agreement between the predicated scores and the ground truth labels. The score function is a linear mapping, that maps a single input data, in the form of a vector if the data has multiple features, to a scalar. Here it is $f(x_i, w, b) = wx_i + b$, where $w$ is the weights and $b$ is the bias, and for multiple labeled classes, each will have different set of weights and bias to calculate the scores respectively. Besides, this also can be interpreted as dot product of the weights and the input data, which each weight vector can be considered as a template for one of the classes. With this terminology, the linear classifier is doing template matching, where the "templates" are learned.

Regarding learning, there are 2 loss functions I've learnt so far, the multiclass SVM loss and the softmax classifier with cross-entropy loss (which I will leave it out of this blog). The basic idea of SVM loss is that it expects the correct class for each input data to have a score higher than the other incorrect classes by some fixed margin. And this expectation is expressed using this loss function $L_i = \sum_{j \neq y_i} \max\left(0, s_j - s_{y_i} + \Delta\right)$, where $L_i$ is the loss for the $i-th$ input data. $s_j$ is the score calculated for class $j$, $s_{y_i}$ is the score for the true class $y_i$, $\Delta$ is the margin hyperparameter (typically $\Delta = 1$), and $max(0, -)$ is the Hinge function (activates only when $s_j - s_{y_i} + \Delta > 0$). In practice, there is also a L2 regularization loss added to the function, which prefers smaller and more diffuse weights. This effect can improve the generalization performance and lead to less overfitting. The optimization algorithm to minize this loss function is using gradient descent, which is also used in backpropagation for training neural network.

> As a new beginner, with little prior knowledge of SVM, I was somewhat confused with multiclass SVM loss (linear classifier) and kernelized SVM (non-linear classifier). After some search online and diverge the learning path a bit towards support vector machine, I get some basic understanding. The multiclass SVM loss is just a training objective (to minimize) for linear models, and the loss itself doesn't create non-linearity but the learning model architecture does. Thus, multiclass SVM loss pairing with linear model can only learn linear boundaries, but if pairing with neural network which has hidden layers with non-linear activation, e.g. ReLU, Sigmod, etc., it can learn non-linear boundaries and complex patterns. Whereas, kernelized SVM uses kernel to map the non-linear data to higher dimensions that can be linearly separated by the supported vectors. Thus it is a powerful tool to handle non-linear data. I will leave this topic for now, and come back later for a better understanding mathematically.
>

# What is Convolutional Neural Network

# The Key Points I am Concerning the most

# My Learning Principles
