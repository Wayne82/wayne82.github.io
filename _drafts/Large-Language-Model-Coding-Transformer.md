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

## The Transformer Model Architecture in Details
At its core and a very high level, the Transformer model is to predict the **next token** in a sequence based on the **preceding tokens**. Simply it can be illustrated as below figure, considering the transformer model as a magic function that takes in a sequence of tokens and outputs the next token, and then appends the new token to the input sequence for yet another new token prediction, and so on.

![Transformer Simplified View](/assets/images/transformer.simplified.png)

This is achieved through a stack of Transformer blocks, each consisting of two main components: Multi-Head Self-Attention and Feed-Forward Neural Networks, along with Layer Normalization and Residual Connections. The details of the architecture can only be better understood by diving into the code implementation, and here is a breakdown of the key components inside the Transformer model:

![Transformer Model Architecture](/assets/images/transformer architecture in details.png)

The code implementation of this basic Transformer model for my own education purpose can be found in my [nn-learn repo](https://github.com/Wayne82/nn-learn/blob/main/gpt_transformer.py), and there are around 200 lines of code implementing the model structures shown in the above figure. The code can be such concise because of the modular design of PyTorch, which provides many built-in functions and classes, and then building the Transformer model becomes a matter of assembling these components together, like lego blocks. Below is a brief explanation of the major processing steps inside the model:

* We start with a sequence of input texts.
* The input texts are first tokenized into discrete tokens, a list of integers representing words or subwords.
* Each token is then converted into a vector representation through an embedding layer.
* Positional encodings are added to the token embeddings to retain information about the sequence order.
* The combined embeddings are passed through multiple Transformer blocks in sequence.
* Each Transformer block applies multi-head self-attention to capture relationships between tokens, followed by a feed-forward neural network to process the information further.
* Layer normalization and residual connection are used inside each Transformer block to stabilize training process and improve gradient update flow.
* Each head in the multi-head block uses the attention mechanism to compute attention scores by scaled dot-product of queries and keys, applies a mask to prevent attending to future tokens, and then finally produces weighted sums of values based on these scores.
* Output from each head is concatenated and projected back to the original embedding dimension.
* A linear layer maps the output of the last Transformer block in embedding space to vector of logits in the vocabulary size dimension for each token.
* Finally, the logits are converted to probabilities of every possible token using softmax,
  * For training, the model computes the cross-entropy loss between predicted probabilities and the expected next token, and then updates the model parameters using backpropagation and an optimizer (AdamW in this case). Besides, training is done in batches of input sequences to improve efficiency.
  * For inference, the model samples the next token using Weighted Random Sampling based on the predicted probabilities.

> 📝 Notes (What I quoted from Andrej Karpathy's viedo courses)
>
> * Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
> * There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
> * Each example across batch dimension is of course processed completely independently and never "talk" to each other.
>

## Data Preparation

## Training Results

## Overfitting and The Scaling Laws

## Conclusion and Next Steps