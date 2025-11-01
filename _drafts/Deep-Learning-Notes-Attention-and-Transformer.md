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
In my previous post on [Recurrent Neural Networks and BPTT](http://localhost:4000/neural-network/2025/08/03/Neural-Network-Notes-Recurrent-Neural-Network-and-BPTT.html), I had this question - `"How did the development of RNNs lead to attention mechanisms, Transformers, GPT, and modern LLMs?"`, and after a few weeks of learning with various resources, I gradually gained a general picture of the evolution from RNNs to attention mechanisms, and further to Transformer architectures.

* RNNs were designed to handle sequential data back in the 1980s. Instead of treating each input independently, it maintains a **hidden state** that captures information about previous inputs. However, RNNs suffer from issues like **vanishing gradients**, making it difficult to learn long-range dependencies in sequences.
* To mitigate the vanishing gradient problem, **LSTM and GRU** architectures were introduced in late 1990s and early 2010s. They can capture longer-term dependencies better than vanilla RNNs, but they are still sequential in nature - can't parallelize well during training.
* Later when tasks like machine translation became popular, a sequence to sequence (**seq2seq**) model based on RNNs was proposed, which enables end-to-end learning from input to output, and achieves high performance. However, it still struggles with long sequences, and the single “**context vector**” becomes a bottleneck — it forces the entire meaning of a long sentence into one fixed-size vector.
* To solve the “**context vector**” bottleneck, **attention** was introduced in the Seq2Seq model! Attention allows the model to dynamically **attend to** different parts of the input sequence when generating each part of the output sequence. This means that instead of relying on a single context vector, the model can attend to relevant parts of the input as needed, greatly improving performance on tasks like translation. Until then, this attention idea still relies on RNNs as the backbone architecture, so it still has the sequential processing limitation.
* Time advanced to 2017, the **Transformer** architecture was proposed in the landmark paper "Attention is All You Need". This was the revolutionary leap - it uses **attention without recurrence** at all!

## The Attention Mechanism
I find the Stanford CS231N Lecture 8 on Attention Mechanism and Transformers very helpful to understand the attention mechanism conceptually and how it is developed from RNNs.

### Attention in Seq2Seq RNNs
This diagram below illustrates the attention mechanism initially introduced in the seq2seq model with RNNs to address the single context vector bottleneck:
![Attention in RNN](/assets/images/attention%20in%20rnn.png)
* Now each step in the decoder can attend to different parts of the input sequence by introducing the context vector \(c_t\).
* The context vector \(c_t\) is computed as a weighted sum of the encoder hidden states \(h_i\).
* The weights \(alpha_{t,i}\) are the (scalar) alignment scores that are computed by applying a "linear layer" to the concatenation of one of the decoder hidden states and the encoder hidden states, followed by a softmax function to normalize the alignment scores to get the attention weights.
* The "linear layer" can be implemented in various ways, such as using the **Bahdanau attention** (additive), **Luong attention**, or **dot-product attention** (multiplicative).

### General Attention Mechanism
Then the attention mechanism turns out to be very powerful computational primitive for neural networks in its own right, and can be used without RNNs at all. It is a general operator that takes in 3 set of vectors: **queries (Q)**, **keys (K)**, and **values (V)**, and computes a weighted sum of the values, where the weights are determined by the similarity between the queries and keys.
* The inputs can be query vectors and data vectors in the cross-attention case, or just one set of input vectors in the self-attention case.
* The learnable weight matrices are the key matrix, value matrix, and query matrix in the self-attention case.
* The outputs can be calculated using scaled dot-product attention method:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Below are the different types of attention layers I've captured from the lecture:

![Cross Attention](/assets/images/cross%20attention%20layer.png)
* This is called **cross-attention**, where it has 2 sets of inputs - one is the query vectors, and the other is the data vectors which are used to project to keys and values.

![Self Attention](/assets/images/self%20attention%20layer.png)
* This is called **self-attention**, where it has only one inputs, but project to 3 different things - queries, keys, and values.

![Masked Self Attention](/assets/images/masked%20self%20attention%20layer.png)
* This is called **masked self-attention**, where it is used in predicting the next token in a sequence. The mask ensures that the model can only attend to previous tokens and not future ones.

![Multiheaded Self Attention](/assets/images/multiheaded%20self%20attention%20layer.png)
* This is called **multi-headed self-attention**, where multiple attention heads can be run in parallel, allowing the model to learn different types of relationships simultaneously.

## Transformer Architecture
Now, I can read the original paper "[Attention is All You Need](https://arxiv.org/pdf/1706.03762)", much easier than the first attempt a few years ago. The main breakthrough of the Transformer architecture is that it relies solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Besides, there are also some other key components in the Transformer architecture, taking important roles. Put them together comes the whole picture (image is referenced from the paper):
![Transformer Architecture](/assets/images/transformer%20architecture.png)
* The original Transformer model consists of an encoder and a decoder, each composed of multiple identical layers.
* Each encoder layer has two sub-layers: a multi-headed self-attention component and a fully connected feed-forward network.
* Each decoder layer has three sub-layers: a masked multi-headed self-attention component and a multi-headed cross-attention component, followed by a fully connected feed-forward network.
* Residual connections and layer normalization are applied after each sub-layer to stabilize training.
* Positional encoding is added to input embeddings for both the encoder and decoder to provide information about the position of tokens in the sequence, since the recurrent structure with the natural sequence order is removed completely.
* The word embeddings are part of the transformer model itself, that are learned during the training process. This means there isn't a pre-existing, separate word embedding table (like Word2Vec or GloVe) that the model starts with. Instead, the embedding vectors for every word in the vocabulary are parameters of the Transformer model and are learned from scratch simultaneously with the rest of the model's weights during the training process.
* For a sequence of n tokens input, each head in the multi-headed attention layer products n output vectors - one per token. Then, these outputs from all heads are concatenated along the same token positions, and projected to produce the final n output vectors of the multi-headed attention layer. The mathematical formulation is as follows:

  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$

  $$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

  where, $X$ is the input sequence, and $W_i^Q$, $W_i^K$, $W_i^V$, $W^O$ are learnable weight matrices.

* At the end of the Transformer model after going through all layers, we get n contextual enriched embeddings corresponding to the n input tokens. Each embedding will be individually passed through a linear layer followed by a softmax function:

  $$ P(w_{i+1}|w_1, w_2, \ldots, w_i) = \text{softmax}(E_i W^P + b^P) $$

  which gives n probability distributions of next token for each input token.

* Finally, when using the pre-trained Transformer model to predict the next token in a sequence, we only take the last token's output probability distribution to produce it.

## Learning Resources
- [Stanford CS231N \| Lecture 8: Attention Mechanism and Transformers](https://www.youtube.com/watch?v=RQowiOF_FvQ&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=8&pp=iAQB)
- [Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy](https://jalammar.github.io/illustrated-transformer/)

## Conclusion and Next Steps