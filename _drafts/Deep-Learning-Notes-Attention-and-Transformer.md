---
layout: post
title:  "Deep Learning Notes: Attention and Transformer"
categories: neural-network
mathjax: true
comments: true
---

Eventually, my journey through neural networks has brought me to a point where I feel equipped with enough foundational knowledge to delve into the **attention mechanism** and **transformer models** in deep learning. I can’t help but start this post to record what I’ve learned so far, even as I continue exploring and deepening my understanding of this intriguing subject.

## Look Back Where I Started
Since this is a big milestone for me where I feel confident in understanding the 2 advanced topics, I want to take a moment to reflect on how I got here.

The first time I encountered the term "attention mechanism" and "transformer" was when chatGPT was released by OpenAI in November 2022. It caught everyone's attention, because it was a breakthrough in NLP and demonstrated impressive capabilities in generating human-like texts/responses. Particularly, it aroused my great interest and make me wonder how it worked behind the chat box. At that time, I have no idea about the terms of "Generative Pre-trained Transformer" from the name chatGPT itself, how it is related to the attention mechanism, and what is the famous "Attention is All You Need" paper about.

I know it is not an easy task to understand all the related techniques in depth, for someone like me who has no prior knowledge in NLP and deep learning, but has strong curiosity to dig into the mathematical level of details until I feel satisfied. It is about 2 years later, in the beginnig of this year 2025, I finally decided to take the challenge and started this learning journey.

My main approach is to start from the basics of neural networks, and follow the trace of deep learning advancements to learn the key techniques along the way and step by step ([Neural Network Basics and Backpropagation](https://wayne82.github.io/neural-network/2025/03/30/Neural-Network-Notes-The-Basics-and-Backpropagation.html), [Convolutional Neural Networks](https://wayne82.github.io/neural-network/2025/05/15/Neural-Network-Notes-Convolutional-Neural-Network.html), [Recurrent Neural Networks](https://wayne82.github.io/neural-network/2025/08/03/Neural-Network-Notes-Recurrent-Neural-Network-and-BPTT.html), and [NLP - Word2Vec Embedding](https://wayne82.github.io/neural-network/2025/10/12/Natural-Language-Processing-Notes-Word2Vec.html)), until I reach the attention mechanism and transformer models naturally.

Now here I am.

## From RNN to Attention
In my previous post on [Recurrent Neural Networks and BPTT](http://localhost:4000/neural-network/2025/08/03/Neural-Network-Notes-Recurrent-Neural-Network-and-BPTT.html), I had this question - "How did the development of RNNs lead to attention mechanisms, Transformers, GPT, and modern LLMs?", and after a few weeks of learning with various resources, I gradually gained a general picture of the evolution from RNNs to attention mechanisms, and further to Transformer architectures.

* RNNs were designed to handle sequential data back in the 1980s. Instead of treating each input independently, it maintains a **hidden state** that captures information about previous inputs. However, RNNs suffer from issues like **vanishing gradients**, making it difficult to learn long-range dependencies in sequences.
* To mitigate the vanishing gradient problem, **LSTM and GRU** architectures were introduced in late 1990s and early 2010s. They can capture longer-term dependencies better than vanilla RNNs, but they are still sequential in nature - can't parallelize well during training.
* Later when tasks like machine translation became popular, a sequence to sequence (**seq2seq**) model based on RNNs was proposed, which enables end-to-end learning from input to output, and achieves high performance. However, it still struggles with long sequences, and the single “**context vector**” becomes a bottleneck — it forces the entire meaning of a long sentence into one fixed-size vector.
* To solve the “**context vector**” bottleneck, **attention** was introduced in the Seq2Seq model! Attention allows the model to dynamically **attend to** different parts of the input sequence when generating each part of the output sequence. This means that instead of relying on a single context vector, the model can attend to relevant parts of the input as needed, greatly improving performance on tasks like translation. Until then, this attention idea still relies on RNNs as the backbone architecture, so it still has the sequential processing limitation.
* Time advanced to 2017, the **Transformer** architecture was proposed in the landmark paper "Attention is All You Need". This was the revolutionary leap - it uses **attention without recurrence** at all! The radical ideas introduced in this paper include:
  - **No recurrence at all** - Instead of processing sequences step by step, the Transformer processes the entire sequence **in parallel** using **self-attention** mechanisms. This allows for much faster training and better scalability using GPUs.
  - **Positional encoding** - Since the Transformer does not have a built-in notion of sequence order (unlike RNNs), it uses positional encodings to inject information about the position of each token in the sequence.
  - **Multi-head attention** - Multiple attention heads allow the model to learn different types of relationships simultaneously (e.g., syntactic, semantic, positional).
  - **Layer normalization and residual connections** - To stabilize and accelerate training of very deep attention stacks.

## The Attention Mechanism

## Transformer Architecture

## Learning Resources
- [Stanford CS231N \| Lecture 8: Attention Mechanism and Transformers](https://www.youtube.com/watch?v=RQowiOF_FvQ&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=8&pp=iAQB)
- [Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy](https://jalammar.github.io/illustrated-transformer/)

## Conclusion and Next Steps