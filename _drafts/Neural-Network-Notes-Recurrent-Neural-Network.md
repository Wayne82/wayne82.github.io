---
layout: post
title:  "Neural Network Notes: Recurrent Neural Network and BPTT"
categories: neural-network
mathjax: true
comments: true
---

Until now, I am quite happy with the progress of my neural network learning journey. I have covered the [basics of neural networks](https://wayne82.github.io/neural-network/2025/03/30/Neural-Network-Notes-The-Basics-and-Backpropagation.html), [convolutional neural networks](https://wayne82.github.io/neural-network/2025/05/15/Neural-Network-Notes-Convolutional-Neural-Network.html), and also get my hands on coding the [vanilla neural network](https://wayne82.github.io/neural-network/2025/06/01/Neural-Network-Notes-Coding-the-Network-from-Scratch.html) and [convolutional neural network](https://wayne82.github.io/neural-network/2025/07/12/Neural-Network-Notes-Coding-CNN.html) from "scratch". Now, it is time to dive into another innovative architecture: **recurrent neural networks** (RNNs), a type of artificial neural network designed to process sequential data, where the order of elements is crucial, like text or time series.

> 📝 Notes
>
> Just a kind reminder to those who occasionally come across my blog: I'm not an expert in neural networks—I'm still learning. The notes here and in the [entire series](https://wayne82.github.io/topics/neural-network/) are mainly a record of my personal learning journey. That said, I'm glad if any part of them is helpful to you as well.

## What is a Recurrent Neural Network?
A Recurrent Neural Network (RNN) is a type of neural network specialized for processing sequences, where the order of inputs matters — such as in text, speech, or time-series data. Unlike traditional feedforward networks, RNNs introduce recurrent connections, meaning the output at each time step is fed back into the network to influence future predictions. This feedback loop allows the model to retain a form of memory, making it suitable for tasks involving context and temporal dependencies.

When I first encountered this definition, I found it quite abstract. What exactly does it mean to "feed back" outputs? What does the architecture really look like? These questions, and more, came to mind:

* What does the **architecture of a vanilla RNN** look like, and how do the **recurrent connections** work?
* How is **backpropagation** applied in RNNs, and why are they more prone to the **vanishing gradient problem**?
* How are **RNNs trained** for tasks like sequence prediction or text generation?
* How do we use a trained RNN model to **generate new text given a seed?**
* Can I **implement a simple RNN** from scratch using just Python and NumPy?
* What are the main RNN variants like **LSTM and GRU**, and how do they improve over vanilla RNNs?
* How did the development of RNNs lead to **attention mechanisms, Transformers, GPT, and modern LLMs?**

These are the questions I aim to figure out. Since the topic is quite broad, it can expand into multiple posts. Here’s a rough outline of what I plan to cover:

* This post focuses on understanding the vanilla RNN architecture and backpropagation.
* The next can dive into a hands-on implementation of a simple RNN from scratch, using Python and NumPy.
* The third may look at RNN variants and their practical use cases.

After that, I hope to be well-prepared to dive deeper into the world of attention, transformers, and modern LLMs.

## A vanilla RNN Architecture
The simplest form of an RNN architecture consists of an input layer, a hidden layer with recurrent connections, and an output layer. The key differentiation from traditional neural networks is that the hidden layer's output at each time step is fed back into the same hidden layer for the next time step, allowing the network to maintain a form of memory by incorporating information from previous time steps.

See the figure below:

![Vanilla RNN Architecture](/assets/images/vanilla%20RNN%20architecture.png)

Usually, the architecture can also be represented in an unrolled form, where each time step is shown separately, making it easier to visualize the flow of data input through time.

![Unrolled RNN Architecture](/assets/images/RNN%20unrolled%20architecture.png)

It is also worth to expand the simple form of RNN architecture to show neurons for each layer in its compact form, which can illustrate a direct comparison with traditional feedforward networks.

![RNN Neurons](/assets/images/RNN%20neurons.png)

The mathematical expression for the RNN can be written as follows:

$$\begin{align}
h_t &= f(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \\
y_t &= W_{hy} h_t + b_y
\end{align}$$

where:
- **$$h_t$$** is the hidden state at time $$t$$
- **$$x_t$$** is the input at time $$t$$
- **$$y_t$$** is the output at time $$t$$
- **$$W_{hh}$$** is the weight matrix from hidden to hidden (recurrent weights)
- **$$W_{xh}$$** is the weight matrix from input to hidden
- **$$W_{hy}$$** is the weight matrix from hidden to output
- **$$b_h$$** and **$$b_y$$** are the biases for the hidden and output layers, respectively
- **$$f$$** is the activation function (commonly tanh or ReLU)

## Backpropagation Through Time (BPTT)

## Vanishing Gradient Problem

## My Learning Process and Looking Ahead

## References