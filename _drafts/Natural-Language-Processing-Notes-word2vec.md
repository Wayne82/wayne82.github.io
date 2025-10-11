---
layout: post
title:  "Natural Language Processing Notes: Word2Vec"
categories: neural-network
mathjax: true
comments: true
---

Now, I’ve set off on a new learning journey into Large Language Models (LLMs). As always, I prefer to begin with the fundamentals and gradually build up my understanding. Recently, I came across the Stanford course — [CS224N: NLP with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) — available on YouTube. After going through its syllabus, I found it to be an excellent resource for learning the core concepts of Natural Language Processing (NLP) and for preparing myself for more advanced topics related to LLMs.

The course begins with an introduction to the Word2Vec model — a foundational technique for learning word embeddings, which are numerical vector representations of words that capture their semantic meaning and relationships. It is quite a fundamental concept in NLP, so that I spent some extra time going through the details, and eventually felt it would be worthwhile to write down my learning notes as a separate blog post.

## Distributional Hypothesis
The Word2Vec model is based on the **Distributional Hypothesis**, which posits that words appearing in similar contexts tend to have similar meanings. This idea is often summarized by the phrase "**You shall know a word by the company it keeps**", originally coined by linguist J.R. Firth.

## What is Word2Vec Model and Why it is Developed
Word2Vec was developed to create dense vector representations of words, known as word embeddings, that capture semantic and syntactic relationships, allowing machines to understand word meanings more effectively.

Traditional methods like **one-hot encoding** represent words as sparse vectors, which do not capture relationships between words. For example, in one-hot encoding, the words "king" and "queen" would be represented as orthogonal vectors, failing to reflect their semantic similarity.

There are two main architectures in the Word2Vec model:
1. **Continuous Bag of Words (CBOW)**: This architecture predicts a target word based on its surrounding context words. It takes the average of the context word vectors and uses that to predict the target word.
2. **Skip-Gram**: This architecture does the opposite of CBOW. It predicts the surrounding context words given a target word. It uses the target word vector to predict each of the context words.

The goal of the 2 architectures is the same, which is to learn the word embeddings using a **shallow neural network**, which are trained using large corpora of text data.

## The Neural Network Structure
The shallow neural network used in Word2Vec consists of 3 layers:
1. **Input Layer**: This layer represents the input word (or context words in CBOW). Each word is represented as a **one-hot encoded vector**.
2. **Hidden Layer**: This layer contains the weights that will be learned during training. The size of this layer determines the dimensionality of the word embeddings.
3. **Output Layer**: This layer produces a probability distribution over the vocabulary, predicting the target word (or context words in Skip-Gram).

This shallow neural network took me a bit of time to reason about, as it is a little bit different from the typical neural networks, in below aspects:

* There is no activation function between the input layer and the hidden layer, and a **softmax activation function** is applied at the output layer to convert the logits into probabilities.
* The inputs for CBOW are average of **multiple context word vectors**, while for Skip-Gram, the input is the **only one target word vector**.
* The outputs for CBOW is the **only one target word**, while for Skip-Gram, the outputs are **multiple context words**.
* Once the training is done, the word embeddings are obtained from the weights between the input layer and the hidden layer as the final results. And then, the entire neural network is no longer needed and there is no further prediction to be made either.

## Key Notes

## What is Next?