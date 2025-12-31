---
layout: post
title:  "Large Language Model Notes: Coding Transformer with PyTorch"
categories: neural-network
mathjax: true
comments: true
---

In my [previous post](https://wayne82.github.io/neural-network/2025/11/02/Large-Language-Model-Notes-Attention-and-Transformer.html), I summarized my learning experience and understanding of the Attention mechanism and Transformer architecture from a high-level, theoretical perspective. Now, it is time to get hands-on.

This post records my journey of building a basic Transformer model from scratch in PyTorch. By training this model on classic Chinese poetry, I aimed to move beyond theory and gain a deep, practical understanding of how Large Language Models (LLMs) are actually built and trained.

What surprised me most was the conciseness of the implementation. Thanks to the modular design of PyTorch and the elegance of the Transformer architecture itself, it is possible to build a functional, GPT-like model with only around 200 lines of code!

## Where I Started
As always, I began my learning journey by seeking out the best online courses, tutorials, and official documentation. I was incredibly grateful to discover Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series on YouTube. This series provides a crystal-clear, well-structured explanation of the entire landscape — from the basic concepts of neural networks and backpropagation to bigram language models, the attention mechanism, and finally, the implementation of a GPT-like Transformer in PyTorch.

Although the videos are long, Karpathy explains complex concepts in a highly concise and intuitive way. He also shares numerous coding details and tricks that are invaluable for beginners looking to get started quickly. It is also worth noting that my previous studies on neural network fundamentals laid a solid foundation, allowing me to follow his video courses closely without frequent pauses.

## The Transformer Model Architecture in Depth

## Data Preparation

## Training Results

## Overfitting and The Scaling Laws

## Conclusion and Next Steps