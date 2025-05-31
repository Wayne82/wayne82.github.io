---
layout: post
title:  "Neural Network Notes: Coding the Network"
categories: neural-network
mathjax: true
comments: true
---

I’ve set out on a journey to learn and understand AI — with the ultimate goal of grasping the essence of large language models (LLMs) and exploring the frontier of research in this field freely. This pursuit is driven by a deeper curiosity: a desire to understand the origins of human consciousness. I believe that advances in AI and the ongoing quest to develop artificial general intelligence (AGI) can offer valuable insights into this profound question.

To begin this journey, I started by studying [the fundamentals of neural networks and the backpropagation algorithm](https://wayne82.github.io/neural-network/2025/03/30/Neural-Network-Notes-The-Basics-and-Backpropagation.html), followed by [an introduction to convolutional neural networks](https://wayne82.github.io/neural-network/2025/05/15/Neural-Network-Notes-CNN.html).

Before moving further, I want to solidify my understanding by getting hands-on — coding a basic neural network with the backpropagation algorithm from scratch. I’ll be using and learning from the excellent code example in the online book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits). My goal is to truly grasp the core mechanism that powers a neural network’s ability to learn.

Then, I've started this github repo - [nn-learn](https://github.com/Wayne82/nn-learn) for managing my code implementation of the learning. And this blog is to document the experience of coding a basic neural network along the way.

## The Structure of the Code
Surprisingly, fewer than 100 lines of python code are enough to build a basic neural network architecture, implement the training loop, and update weights and biases effectively using stochastic gradient descent (SGD) combined with the backpropagation algorithm.

After just a few dozens training iterations, the network is already able to predict handwritten digit images with impressive accuracy — often exceeding 95%.

The diagram below provides a visual overview of how the code is structured:

![Image](/assets/images/neural%20network%20code%20structure.png)

To be clear, I didn’t write all of this entirely from scratch. I referred to the [code example](https://github.com/mnielsen/neural-networks-and-deep-learning) from the excellent online book Neural Networks and Deep Learning by Michael Nielsen (mentioned earlier). My modified version of the implementation can be found in [this file](https://github.com/Wayne82/nn-learn/blob/main/nnet.py) in my GitHub repo, where I’ve added extra comments to make the code more self-explanatory.

One part I want to highlight here is the backpropagation function. I’ve annotated each step of the code with references to the corresponding equations from the book:
```javascript
        def _backprop(self, y):
        """
        Backpropagation algorithm to compute gradients.
        :param y: Target output
        """

        """
        Calculate the gradient of the cost function with respect to
        the output of the network.
        Apply equation (BP1)
        """
        delta = self.cost_fn.prime(y, self.activations[-1]) * \
                self.activation_fns['output'].prime(self.zs[-1])
        """
        Apply equation (BP3) and (BP4)
        """
        self.nabla_b[-1] += delta
        self.nabla_w[-1] += np.dot(delta, self.activations[-2].T)

        """
        Iterate through the layers in reverse order to compute gradients
        """
        for l in range(2, self.size):
            z = self.zs[-l]
            sp = self.activation_fns['hidden'].prime(z)

            """
            Apply equation (BP2)
            """
            delta = np.dot(self.weights[-l + 1].T, delta) * sp

            """
            Apply equation (BP3) and (BP4)
            """
            self.nabla_b[-l] += delta
            self.nabla_w[-l] += np.dot(delta, self.activations[-l - 1].T)
```
The four fundamental equations used above are explained in detail in [the book](http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation) and were also discussed in [my earlier blog post](https://wayne82.github.io/neural-network/2025/03/30/Neural-Network-Notes-The-Basics-and-Backpropagation.html), where I gave the complete proof.

![Image](/assets/images/equations%20of%20backpropagation.png)

## Challenges Along the Way

## Excitment to See - It Works!

## Exploring Activation Functions and L2 Regularization

## This is Just a Beginning
