---
layout: post
title:  "Neural Network Notes: The Basics and Backpropagation"
categories: neural-network
mathjax: true
---

As an individual, I feel incredibly lucky to live in an era largely free from major national or global wars, while both fundamental sciences and engineering are advancing rapidly, reaching new peaks one after another at an unprecedented pace. Among these breakthroughs, Artificial Intelligence — particularly Large Language Models (LLMs) — stands at the forefront, sparking discussions about AI, AGI, and the future of intelligence. This technology is both fascinating and revolutionary — not only because of the groundbreaking moment when LLMs first enabled fluent human-machine conversations, making what once seemed like science fiction a reality, but also due to their rapidly expanding capabilities, which have the potential to transform nearly every industry.

Out of deep curiosity, I set out boldly to understand the underlying techniques that led to these breakthroughs and to explore whether any fundamental principles of intelligence have been uncovered. But, the first step for me is to review the basics of neural networks and grasp one of the core algorithms — backpropagation. This blog serves primarily as a record of my own learning notes, rather than a comprehensive explanation on this subject.

# Neural Network Basics
Briefly, a neural network is a machine learning model inspired by the structure and functioning of human brain, composed of interconnected nodes (neurons) that process sample data through weighted connections in layers, allowing it to learn and make predictions on new data. It is widely used for tasks such as image recognition, natural language processing, and so on.
## Key Concepts
As what I have learned so far, the key concepts come to me include,
* **Layered structure**, a typeical neural network consists of the following layers:
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
* **Gradient Descent**, is the algorithm used to update the weights and biases of the neural network to minimize the cost function. The update equation for a weight $w$ is given by: $$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$ where,

  $w$: the weight being updated.

  $\eta$: the learning rate, a positive number controlling the step size of the update.

  $\frac{\partial L}{\partial w}$, the partial derivative of cost function $L$ with respect to weight $w$. And here the cost function $L$ include the entire training dataset.
  * **Stochastic Gradient Descent (SGD)**, instead of using the entire dataset, SGD randomly selects a small subset of the dataset, called a **mini-batch** to compute the cost function $L$.
* **Backpropagation**, is the fundamental algorithm used to train neural networks. It efficiently computes the gradients of the cost function with respect to weights and biases by applying the **chain rule of calculus**. These gradients are then used to update the weights and biases via **gradient descent**, and it starts from the output layer and moving **backwards** to the input layer.
* **Training**, is the process of iteratively updating a neural network's weights and biases to minimize the cost function until it can't improve further.
  * **Epoch**, an epoch is one complete pass through the entire training dataset.
  * **Batch**, a batch is a subset of the training data used in one forward and backward pass.
  * **Iteration**, an iteration is one forward calculation of output and backward update of weights and biases using a single batch of training data.
* **Hyperparameters**, these are the parameters that are set before training a neural network. Unlike the weights and biases parameter, which are learned during training, hyperparameters are not learned from the data but are chosen by the designer. They can significantly impact the performance of the neural network. Some of typical hyperparameters include **Learning Rate**, **Batch Size**, **L2 Regularization Parameter**, etc.
* **Training Dataset**, the dataset is typically split into 3 groups,
  * **Training Set**, used to train the model.
  * **Validation Set**, used to tune hyperparameters and monitor the model's performance during training.
  * **Test Set**, used to evalaute the final performance of the neural network after training.

## The Architecture of Neural Network
A vanilla neural network is the simplest and most fundamental type of neural network. Its architecture can be defined as,
* Consists of **input layer**, one or more **hidden layer**, and an **output layer**.
* Every neuron in one layer is connected to every neuron in the next layer (**fully connected** layers).
* Data flows in one direction - **feedforward**, from input to output, with no loops.

And below diagram illustrate this architecture,
![Image](/assets/images/neural%20network%20architecture.png)
## Learning with Gradient Descent
Now, we have a basic neural network as illustrated above, then how can it learn from training dataset and predict well enough on new data input? The goal of the training process is to find the weights and biases so that the output from the network approximates the expected output y for all training dataset input x. To quantify how well we can achieve this goal, we can define the quaratic cost function, 

$$
\begin{eqnarray}
C(w,b)=\frac{1}{2n} \sum_x \| y(x) - a\|^2
\end{eqnarray}
$$

where, $y(x)$ is the expected output vector and $a$ is the vector of outputs from the network for input $x$. And the cost $C$ is the average of the sum over all $n$ training dataset. The additional coefficent $1/2$ is mainly a convenience for cancelling out the multiplication of number $2$ after derivative. 

Then, the goal of the training is to minimize this cost function so that $C(w, b) \rightarrow 0$. The good news is that this cost function is a smooth or continous function which making small changes in the weights and biases can get effective improvement in the cost. And the changes of the cost with respect to small changes to $w$ (biases $b$ is the same) is given by 

$$
\Delta C \approx \nabla C \cdot \Delta w
$$

where, $\Delta w = (\Delta w_1,\ldots, \Delta w_m)^T$ is the vector of changes made to each weight, $\nabla C$ is the gradient vector with respect to weights,

$$
\begin{eqnarray}
  \nabla C \equiv \left(\frac{\partial C}{\partial w_1}, \ldots, 
  \frac{\partial C}{\partial w_m}\right)^T
\end{eqnarray}
$$

And in order to guarantee the changes to the cost $\Delta C$ is **negative** so that the cost $C \rightarrow  C'=C+\Delta C$ can decrease properly, we can choose the small changes to each weight to be,

$$
\begin{eqnarray}
  \Delta w = -\eta \nabla C,
\end{eqnarray}
$$

where $\eta$ is the learning rate, defining the size or step of the move for a single update for the weights (or biases).

Then, update each weight by this equation,

$$
\begin{eqnarray}
  w \rightarrow w' = w-\eta \nabla C.
\end{eqnarray}
$$

Repeat this update until the cost approximately converge to zero or can't improve further.

In summary, the gradient descent can be viewed as a way of taking small steps in the direction which does the most to immediately decrease $C$.

# Backpropagation
Until now, we saw neural network can learn their weights and biases using the gradient descent algorithm. But how to compute the gradient efficiently is what Backpropagation algorithm come into play.
## Some Preparations
### A Matrix-Based Approach
We need an unambiguous way to mark each weight on the connection from neurons in previous layer to neurons in next layer. And the convention is to use $w^l_{jk}$ to denote the **weight** for the connection from the $k^{\rm th}$ neuron in the $(l-1)^{\rm th}$ layer to the $j^{\rm th}$ neuron in the $l^{\rm th}$ layer. Similarly we use $b^l_j$ for the **bias** of the $j^{\rm th}$ neuron in the $l^{\rm th}$ layer, and we use $a^l_j$ for the **activation** of the $j^{\rm th}$ neuron in the $l^{\rm th}$ layer. See diagram below shows a specific weight, related biases and the activiations.

![Image](/assets/images/nn_labeled.png)

Then with these notations, the activation $a^{l}_j$ of the $j^{\rm th}$ neuron in the $l^{\rm th}$ layer is related to the activations in the $(l-1)^{\rm th}$ layer by below equation (here we use the sigmod activation function)

$$
\begin{eqnarray} 
  a^{l}_j = \sigma\left( \sum_k w^{l}_{jk} a^{l-1}_k + b^l_j \right)
\end{eqnarray}
$$

where the sum is the over all neurons $k$ in the $(l-1)^{\rm th}$ layer. Then, we could write the weights connecting from $(l-1)^{\rm th}$ layer to $l^{\rm th}$ layer as weight matrix $W^l$. Similarly, for each layer $l$ we define bias vector $b^l$, and finally the activations of neurons in the $l^{\rm th}$ layer can be defined as vector $a^l$. The above equation can be rewritten in the compact vectorized form,

$$
\begin{eqnarray} 
  a^{l} = \sigma(W^l a^{l-1}+b^l)
\end{eqnarray}
$$

From the matrix representation of the weights, you can now see why the notation for an individual weight follows the order of the indices $j$ and $k$. This ordering reflects how the weight matrix left-multiplies the activation vector from the previous layer.

### Regard The Neural Network as a Giant Composite Function
The neural network can be considered as a **giant composite function**. This is a fundamental way to mathematically understand neural network. A neural network stacks multiple layers of simplier functions (neurions) to form a nested composition of functions. In a simplified form, for a network with $L$ layers, the output $y$ is computed as:

$$
\mathbf{y} = f_L \left( f_{L-1} \left( \dots f_1(\mathbf{x}) \dots \right) \right)
$$

For a more detailed representation with respect to weights ($W$), biases ($b$), and activation functions ($\sigma$):

$$
\mathbf{y} = \sigma_L \left( \mathbf{W}_L \cdot \sigma_{L-1} \left( \mathbf{W}_{L-1} \cdots \sigma_1(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \cdots + \mathbf{b}_{L-1} \right) + \mathbf{b}_L \right)
$$

It will be mind-blowing to ponder how giant this composite function could be, determined by the depth and width of the neural network.
* **Depth**: modern networks can have hundreds of layers - deep neural network.
* **Width**: each layer may container thousands of neurons.

Then, it could produce millions or even billions of weights and biases that could be adjusted through the training to fine-tune the behavior of the giant composite function. And one of the good properties of this composite function is that it is differentiable (if the activation functions are), which make it possible to learn with gradient descent and use the backpropagation algorithm to fast calculate the gradient.

### Chain Rule of Calculus
Before we dive deep into the actual backpropagation algorithm, I'd like to highlight the chain rule of calculus, which is the foundational mathematical principle that makes backpropagation possbile in neural networks.

Mathematically, the chain rule of calculus can be described as: given a composite function $h$ of 2 differentiable functions $f$ and $g$, such that $h(x) = f(g(x))$, then the derivative of $h$ is calculated as,

$$
h' = f'(g(x)) g'(x)
$$

Then, to remind again the goal of the training of the neural network is to adjust each weight and bias by a small step towards the gradient descent direction. And the key is to calculate the partial derivative of the cost function with respect to each weight and bias. The chain rule breaks this derivative into "composite" parts:

$$
\frac{\partial C}{\partial W^l} = \underbrace{\frac{\partial C}{\partial a^L}}_{\text{Loss gradient}} \cdot \underbrace{\frac{\partial a^L}{\partial z^L}}_{\text{Output activation}} \cdot \underbrace{\frac{\partial z^L}{\partial a^{L-1}}}_{\text{Linear transform}} \cdots \underbrace{\frac{\partial a^{l+1}}{\partial z^{l+1}}}_{\text{Hidden activation}} \cdot \underbrace{\frac{\partial z^{l+1}}{\partial a^l}}_{\text{Linear transform}} \cdot \underbrace{\frac{\partial a^l}{\partial z^l}}_{\text{Activation derivative}} \cdot \underbrace{\frac{\partial z^l}{\partial W^l}}_{\text{Weight gradient}}
$$

Where $L$ is the output layer, $l$ is the index of hidden layers. Then, the key components in this equation:
* $\frac{\partial C}{\partial a^L}$: partial derivative of cost function w.r.t output activations.
* $\frac{\partial a^l}{\partial z^l}$: partial derivative of activation function of layer $l$ w.r.t the weight inputs.
* $\frac{\partial z^l}{\partial a^{l-1}}$: partial derivative of weighted input of layer $l$ w.r.t the activation input from previous layer $l-1$
* $\frac{\partial z^l}{\partial W^{l}}$: partial derivative of weighted input of layer $l$ w.r.t the weights of same layer.

Now, I think I am well prepared to dive deep into next a few sections to uncover the mathematical principles of backpropagation.

## The 4 Fundamental Equations Behind Backpropagation

## Proof of the 4 Fundamental Equations
## The Vanishing Gradient Problems

# Learning Materials

# Final Words
