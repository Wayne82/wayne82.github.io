---
layout: post
title:  "Natural Language Processing Notes: Word2Vec"
categories: neural-network
mathjax: true
comments: true
---

Now, I’ve set off on a new learning journey into Large Language Models (LLMs). As always, I prefer to start from the fundamentals and gradually build up my understanding. Recently, I came across the Stanford course — [CS224N: NLP with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) — available on YouTube. After going through its syllabus, I found it to be an excellent resource for learning the core concepts of Natural Language Processing (NLP) and for preparing myself for more advanced topics related to LLMs.

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
* There are **2 vectors per word** this neural network to learn: one for when the word is an input (learnt as the weights between the input layer and the hidden layer), and another for when the word is an output (learnt as the weights between the hidden layer and the output layer).
* Once the training is done, the word embeddings are obtained from **the weights between the input layer and the hidden layer**, and the weights between the hidden layer and the output layer are usually discarded. Besides, the entire neural network is no longer needed and there is no further prediction to be made either.

## The Loss Functions
Another way to understand the 2 architectures is to look straight at their loss functions respectively, which is a good way to understand the shallow neural network structure as well.

* **CBOW Loss Function**: The loss function for CBOW is the negative log likelihood of the target word given the context words. It can be expressed as:

  $$ L = -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}) $$

  where (w_t) is the center word, and (w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}) are the context words within a window of size (m).

  The probability P(w_t \| w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}) is then computed using the softmax function:

  $$ P(w_t|w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}) = \frac{exp(v_c^T v_o)}{\sum_{w=1}^{V} exp(v_w^T v_o)} $$

  where (v_c) is the vector representation of the center word (w_t), (v_o) is the vector representation of average of the context words (w_{t-m}, ..., w_{t-1}, w_{t+1}, ..., w_{t+m}), and (V) is the vocabulary size.

* **Skip-Gram Loss Function**: The loss function for Skip-Gram is the negative log likelihood of the context words given the target word. It can be expressed as:

  $$ L = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} | w_t) $$

  where (w_t) is the center word, and (w_{t+j}) are the context words within a window of size (m).

  The probability P(w_{t+j} \| w_t) is computed using the softmax function:

  $$ P(w_{t+j}|w_t) = \frac{exp(v_o^T v_c)}{\sum_{w=1}^{V} exp(v_w^T v_c)} $$

  where (v_c) is the vector representation of the center word (w_t), (v_o) is the vector representation of the context word (w_{t+j}), and (V) is the vocabulary size.

  > 📝 Notes
  >
  > The 2 vectors (v_c) and (v_o) used in the 2 loss functions are pairs of vectors for each word, one for when the word is an input (learnt as the weights between the input layer and the hidden layer), and another for when the word is an output (learnt as the weights between the hidden layer and the output layer).

## Key Notes
There are a few other key points I feel worthy to highlight:
* The goal of Word2Vec is **not prediction at runtime — it’s to learn word embeddings**,
  * After training, the entire neural network is not used anymore.
  * What we care about is the geometry of the embedding space (semantic similarity).
  * The word embeddings are obtained from **the weights between the input layer and the hidden layer**, which are sufficient for downstream tasks (classification, clustering, search, etc.).
* Word2Vec only gives **one static embedding per word**, learned from all contexts mixed together.
  * It does not handle polysemy (multiple meanings for the same word).
  * More advanced models like BERT and GPT use context to generate different embeddings for the same word based on its usage (different contexts).

## What is Next?
With this foundation, I feel more confident continuing through the rest of the [CS224N courses](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D). The next exciting topic is the **Attention Mechanism**, which plays a central role in modern NLP architectures such as Transformers and LLMs. I’m looking forward to exploring how it revolutionized the way machines understand and process language.