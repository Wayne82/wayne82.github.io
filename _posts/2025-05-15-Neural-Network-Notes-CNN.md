---
layout: post
title:  "Neural Network Notes: CNN"
categories: neural-network
mathjax: true
comments: true
---

After completing my initial exploration and study of neural networks — and documenting it in [Neural Network Notes: the Basics and Backpropagation](https://wayne82.github.io/neural-network/2025/03/30/Neural-Network-Notes-The-Basics-and-Backpropagation.html) — I am now moving on to the next fascinating topic: Convolutional Neural Networks (CNNs). CNNs are a type of deep learning architecture specifically designed to process data with a grid-like structure, such as images. They excel at tasks like image recognition and classification, and have remained a cornerstone of computer vision and deep learning to this day.

Thanks to the open internet, I was able to access a world-class course — [CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/) — from a top university. After spending about two weeks studying eagerly during my evenings, I completed the course. As usual, I would like to document my notes and learning experiences in this blog.

## Prior Arts
The course is structured into two main modules. The first module covers a substantial amount of foundational content, introducing classic methods for image classification such as **k-Nearest Neighbor (k-NN)** and **linear classification** using **SVM loss** and **Softmax loss**. This naturally leads into an introduction to the **vanilla neural network**. The second module focuses on the main theme: **Convolutional Neural Networks (CNNs)**.

At the core, the image classification problem addressed by all these methods is the task of assigning an input image a label from a fixed set of categories. While this is trivially easy for humans, it is extremely difficult for computers to achieve reliably.

Traditionally, one might try to handcraft an algorithm to distinguish images, but this is no longer practical. Instead, a data-driven approach is adopted: given a well-labeled dataset, the model learns to map images to their corresponding labels through training, and can then predict the class of new, unseen images.

### k-Nearest Neighbor (k-NN)
Different classification methods offer distinct tradeoffs. For instance, the k-NN classifier requires no training time — it simply stores the labeled training data. However, this results in high prediction cost, as it must compare each test image against all training examples.

### Linear Classifier & SVM Loss
In contrast, methods like **Linear Classifiers**, **Neural Networks**, and especially **Deep Neural Networks**, demand significant computation during training but offer efficient prediction once trained.

Linear classifiers consist of two main components:

* **A score function**, which maps input data to class scores. For linear classification, this is typically $f(x_i, W, b) = wx_i + b$, where $W$ is the weight matrix and $b$ is the bias vector.
* **A loss function**, which quantifies the disagreement between predicted scores and the ground truth labels.

Here, each weight vector $w_j$ can be viewed as a learned template for a specific class. This leads to a helpful interpretation: linear classification as template matching.

Among the loss functions introduced, the Multiclass SVM Loss is defined as:

$$
L_i = \sum_{j \neq y_i} \max\left(0, s_j - s_{y_i} + \Delta\right)
$$

where $L_i$ is the loss for the $ith$ input data. $s_j$ is the score calculated for class $j$, $s_{y_i}$ is the score for the true class $y_i$, $\Delta$ is the margin hyperparameter (typically $\Delta = 1$), and $max(0, -)$ is the Hinge function (activates only when $s_j - s_{y_i} + \Delta > 0$).

This function penalizes cases where the correct class score $s_{y_i}$ does not exceed the incorrect class score $s_j$ by at least a margin $\Delta$.

Furthermore, To improve generalization and reduce overfitting, an L2 regularization term is usually added, encouraging smaller weights. The optimization is typically done via gradient descent, which also forms the foundation of backpropagation in neural networks.

> 📝 Notes
>
As a beginner, I was initially confused between Multiclass SVM Loss (used in linear models) and kernelized SVMs (used for non-linear classification). After some detours, I realized that Multiclass SVM Loss is just a training objective and does not introduce non-linearity by itself. The ability to model non-linear boundaries comes from the model architecture, not the loss function. In contrast, kernelized SVMs achieve non-linearity through kernel tricks that implicitly map inputs to a higher-dimensional space, that can be linearly separated by the supported vectors. I will leave the general SVM topic for now, and may come back on this in another time.

### Neural Network
The concept of neural networks is introduced shortly afterwards, with several dedicated sections in the first module of the course. The explanation proceeds by focusing on score functions, avoiding analogies to the human brain. Unlike linear models, neural networks consist of multiple hidden layers. These can be viewed as applying score functions multiple times using different sets of weights, ultimately computing the final scores in the output layer. For example, a three-layers neural network might compute scores as follows:

$$
s = W_3 max(0, W_2 max(0, W_1 x))
$$

where the function $max(0, -)$, used earlier for calculating the multiclass SVM loss, is referred to as the ReLU activation function in the context of neural networks. The non-linear classification capability of neural network arises precisely from such non-linear activation functions.

I won’t delve deeper here, as the foundational concepts of neural networks are covered in my earlier blog post: [Neural Network Notes: the Basics and Backpropagation](https://wayne82.github.io/neural-network/2025/03/30/Neural-Network-Notes-The-Basics-and-Backpropagation.html).

## What is Convolutional Neural Network
A **Convolutional Neural Network (ConvNet or CNN)** is a specialized type of neural network designed specifically for image classification tasks. To recap, a regular neural network has the following architectural properties:
* The input data is treated as a generic, one-dimensional vector of features.
* Neurons in each hidden layer are fully connected to all neurons in the previous layer.
* Connections to neurons are unique—there is no sharing of weights.

In contrast, a ConvNet architecture differs in several key ways by leveraging the spatial structure of image data:
* **3D Layer Volumes**: Since the input consists of images, we can arrange neurons in each layer in three dimensions — **width, height, and depth** — resulting in 3D volumes of neurons.
* **Local Connectivity (Receptive Field)**: Each neuron in a Conv layer is only connected to a small region of the input volume, known as its receptive field, rather than being fully connected.
* **Weight Sharing**: The weights connecting receptive fields to neurons are shared across all neurons in a given depth slice, greatly reducing the number of parameters.

In short, **a ConvNet is composed of layers that each transform an input 3D volume into an output 3D volume via a differentiable function — some with learnable parameters, others not.**

There are three primary types of layers used to build ConvNet architectures:

### Convolutional Layer
The convolutional layer is the core building block of a ConvNet and performs most of the computational heavy lifting.
* The learnable parameters in a Conv layer consist of a set of filters (or kernels). Each filter is small in spatial dimensions (width and height) but extends through the full depth of the input volume. Each neuron connects to a local region in the input volume through a filter.
* There are three key hyperparameters that determine the output volume size:
  * **Depth**: Number of filters used; each filter learns to detect a different feature.
  * **Stride**: The step size used to slide the filter across the input (e.g., a stride of 1 moves one pixel at a time).
  * **Zero-padding**: Padding added around the border of the input to control the spatial size of the output, often used to preserve input size.

  Then, given input volume size **W**, filter size **F**, stride **S**, and zero-padding **P**, the output volume size is  calculated as

  $$
  (W - F + 2P)/S + 1
  $$

* The parameters used in each filter are shared with all neurons in the same depth slice. E.g. a 11x11x3 filter will have $11 \times 11 \times 3= 363$ weights and 1 bias, and shared by all neurons in one depth slice. In AlexNet, the first Conv layer uses 96 such filters, resulting in a total of $11 \times 11 \times 3 \times 96 = 34848$ weights and 96 biases.

### Pooling Layer
The pooling layer is used to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and help to also control overfitting. It is common to periodically insert a pooling layer between successive Conv layers. It operates independently on every depth slice of the input and resize it spatially, using the MAX operation. The most common form is using a filter of size 2 by 2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, effectively discarding 75% of the activations. E.g. a pooling layer with filter size 2, stride 2 will downsample the input volume of size [M x M x D] to output volume of size [M/2 x M/2 x D].
  * A pooling layers introduces **no new parameters**. During backpropagation, gradients are routed to the input element that had the maximum value in the forward pass
  * Some modern architectures suggest omitting pooling layers in favor of Conv layers with larger strides to achieve similar downsampling while preserving parameterization.

### Fully Connected (FC) Layer
The fully connected layer connects every activation in the previous layer to each neuron in the current layer, just like in a standard neural network.

It is possible to convert between convolutional and fully connected layers:
* **Conv to FC**: Any Conv layer can be expressed as an FC layer with a sparse weight matrix, where most entries are zero and non-zero entries are shared due to parameter sharing.
* **FC to Conv**: Any FC layer can be converted into a Conv layer by using a filter with the same spatial dimensions as the input volume, no padding, and stride 1. The output will be a 1×1×D volume — identical to the FC layer output.

### ConvNet Architectures
Finally, ConvNet architectures are built by stacking the above layers — Convolutional, Pooling, and Fully Connected — often with the ReLU activation function explicitly added between layers.
* A common architecture pattern is:
```css
INPUT -> [[CONV -> RELU] * N -> POOL?] * M -> [FC -> RELU] * K -> FC
```
where `*` indicates repetition, and `POOL?` indicates an optional pooling layer. Moreover, `N>=0` and usually `N<=3`, `M>=0`, `K>=0` and usually `K<3`.

* Stacking **smaller Conv filters** is generally preferred over one large receptive field, as it increases non-linearity and reduces the number of parameters while allowing deeper feature extraction. However, it can require more memory during backpropagation to store intermediate activations.
* Some well-known CNN architectures include:
  * **LeNet** (1990s, by Yann LeCun): One of the first successful CNNs, used for digit and zip code recognition.
  * **AlexNet** (2012): Popularized CNNs in computer vision and won the ImageNet competition.
  * **ResNet** (2015–2016): Developed by Kaiming He et al., introduced residual connections and achieved state-of-the-art results.

## My Intrinsic Curiosity Go Further
As I dive deeper into the many aspects and technical details of CNNs — a powerful neural network architecture capable of solving challenging real-world computer vision problems — I find myself continuously pondering these questions:
* How was the multi-layer architecture of CNNs originally developed?
* Why is a convolutional layer commonly interpreted as a feature extractor?
* How is the number of filters in each layer determined?
* How do we ensure that the filters learn distinct and meaningful features, rather than redundant or noisy ones?
* Is there a rigorous mathematical explanation for why this architecture works so effectively for visual data?

I believe these questions reflect the kind of deeper inquiry that often drives innovation and a stronger conceptual understanding. While I don't attempt to answer them in this already lengthy blog post, they remain open areas of curiosity for me. So, I will probably leave my further exploration on these areas to another blog in future.

## My Learning Experience
Whenever I begin learning a new topic, it usually starts with a simple, genuine curiosity about how something works. My learning process tends to follow a pattern. First, I seek out official materials or authoritative books to establish a solid foundation — starting with terminology, key definitions, and core concepts. I then dig into the more detailed mechanics. Along the way, I may refer to high-quality online courses or YouTube videos to deepen my understanding or clarify any confusion.

Once I grasp the basics, I like to go further — exploring why things work the way they do and how they were developed. I find it deeply satisfying to understand the origin and evolution of a subject. Often, these historical or developmental process inspire me to think more broadly and make connections to other domains beyond the topic itself. Then, new insights or ideas may well emerge from these connections involuntarily. Who knows!