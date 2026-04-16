---
layout: post
title:  "How I Learn Diffusion Model"
categories: neural-network
mathjax: true
comments: true
---

It has been over three months since my last post on building a toy Transformer from scratch. At the time, I was torn between two paths: diving deeper into more theoretical knowledge of AI or getting hands-on with real-world projects with agentic AI. Ultimately, I was pulled in the direction of exploring another intriguing generative model: Diffusion Models.
To be honest, it is a daunting topic. It demands a level of mathematical depth that I haven't yet mastered, but to understand the mathematics behind it is just what drives me. I am still in the early stages of this journey, but I want to document and share the things I have learned so far.

## Where I Started
The first time I experienced the "magic" of text to image generation was through the DALL-E model from OpenAI and in the Discord group of Midjourney almost 4 years ago. Though it was very impressive, it didn't trigger my immediate interest, until the Stable Diffusion was released by Stability AI, which also open-sourced the code and model weights. And its the mechanism of creating high-quality images from pure Gaussian noise really caught my deep curiosity.

This was where I started, and finally I decided to spend most of my spare time in the past 3 months learning the diffusion model.

## The Learning Plan
After several days of exploring the web, reading blog posts, and consulting with LLMs, I have converged on a learning path structured into five key stages:

1. **Conceptual Foundations**: This [guest youtube video from 3Blue1Brown](https://www.youtube.com/watch?v=iv-5mZ_9CPY) gives a good general introduction to diffusion models, besides there are also a ton of good resources listed in the video description. Furthermore, this MIT open course [Generative AI - Text-to-Image Models](https://www.youtube.com/watch?v=NQBhhRG-Pe4) provides a slight more technical yet still accessible introduction to this topic, but still at an entry level.
2. **Mathematical Explanation**: This is where the real challenge lies. I initially dove into lilianweng's blog post [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) and the original paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Ho et al. However, I quickly realized I am not equipped with the necessary mathematical knowledge to understand the flow of the explanation or the descriptions in the paper. This led me to this comprehensive online course [MIT 6.S184: Flow Matching and Diffusion Models](https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH) by Peter Holderrieth - a systematic and rigorous resource for understanding the mathematical foundations of flow matching and diffusion models.
3. **Model Architecture**: The next goal is to understand the model architecture, specifically the **U-Nets** and **Diffusion Transformers (DiTs)**, about how they are structured and trained.
4. **Hands-on Implementation**: As usual, after understanding the theoretical aspects well enough, I will try to implement a toy diffusion model from scratch.
5. **Evolution and Research**: Finally, I want to trace the development history of this field to fulfill my curiosity of the process of discovery, and keep up with the latest research papers and advancements in diffusion models, to understand how the field is evolving and where it might be headed in the future.

At this point, I am deep in the second stage. Navigating the mathematical theory behind the model is a big challenge for me, as I had to pause frequently to build the knowledge base of relevant maths in Probability Theory, Stochastic Differential Equations, Multivariate Normal Distribution, Variational Bayesian methods, etc. The goal is not to become a math expert in these areas, but to reach a level where I can understand the mathematical explanations in the original papers and many of the great blog posts out there without much difficulty. I think I am making some progress, but it is still a long way to go.

## What I Understand So Far
The learning notes I took here focus on the specific construction of generative models using flow models, which is trained using flow matching loss.

### The mathematical definition of the problem

Given a set of samples $x_1, x_2, \ldots, x_n$ drawn from an unknown, high dimensional data distribution $p_{data}$, how do we build a generative model that can generate new samples that looks like they are drawn from the same distribution?

### Introduce Flow Models
The generative model can be constructed by flow models. A flow model is described by the Ordinary Differential Equation (ODE):

$$
X_0 \sim p_{init} \tag{random initialization}
$$

$$
\frac{dX_t}{dt} = u_t(X_t) \tag{ODE}
$$

where the ODE is defined by a **vector field $u_t$**. The goal is to make the endpoint **$X_1$** of the flow have distribution **$p_{data}$**. And the learning target is the vector field $u_t$, which is parameterized by the neural network as $u^{\theta}_t$.

Once we have the learned vector field $u^{\theta}_t$, we can then generate new samples by simulating the ODE with Euler method:

$$
X_{t+h} = X_t + hu^{\theta}_t(X_t)
$$

Where $h$ is the step size. By iterating this process from $t=0$ to $t=1$, we can obtain a sample $X_1$ that is approximately drawn from the data distribution $p_{data}$.

### Define Probability Paths
To connect the vector field with the data distribution, we need to define a probability path $p_t(x)$ that evolves over time from the initial distribution $p_{init}$ to the data distribution $p_{data}$. Intuitively, a probability path specifies a gradual interpolation between the initial distribution and the data distribution. One particularly popular choice is the Gaussian propability path. The conditional Gaussian probability path is defined as:

$$
p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 \mathbf{I}_d)
$$

Where $\alpha_t$ and $\beta_t$ are time-dependent noise schedules that control the mean and variance of the distribution at time $t$.

The question now is how can we find a vector field $u_t$ that the flow $X_t$ follow the probability path? Usually, the conditional vector field can be analytically constructed from the conditional probability path. For example, for the Gaussian probability path shown above, the corresponding conditional vector field can be constructed as:

$$
u_t(x|z) = (\alpha_t' - (\beta_t'/\beta_t) \alpha_t)z + (\beta_t' / \beta_t)x
$$

Where $\alpha_t'$ and $\beta_t'$ are the derivatives of $\alpha_t$ and $\beta_t$ with respect to time $t$.

### Marginalization Trick and Continuity Equation
The conditional vector field constructed as above does not help by itself, because all the trajectories of the flow will collapse to the same point $z$, we just regenerate the known data point $z$. However, the conditional vector field serves as a building block for constructing the marginal vector field that can generates actual samples from $p_{data}$. This marginal vector field is defined using the **marginalization trick**:

$$
u_t(x) = \int u_t(x|z) \frac{p_t(x|z) p_{data}(z)}{p_t(x)} dz
$$

Which follows the marginal probability path,

$$
X_0 \sim p_{init}, \quad \frac{dX_t}{dt} = u_t(X_t) \implies X_1 \sim p_{data}
$$

This connection can be proved by the **continuity equation**, which states that the change in probability density over time is equal to the negative divergence of the probability flux:

$$
\frac{\partial p_t(x)}{\partial t} = - \operatorname{div}(p_t u_t)(x)
$$

Where the divergence operator is defined as:

$$
\operatorname{div}(v_t)(x) = \sum_{i=1}^d \frac{\partial v^i_t(x)}{\partial x_i}
$$

### Learning the Vector Field by Flow Matching Loss
Recall that the learning target is the vector field $u^{\theta}_t$, and the loss function is to use a mean-squared error, defined as the **flow matching loss**:

$$\begin{align}
\mathcal{L}_{FM}(\theta)
&= \mathbb{E}_{t \sim \mathcal{U}(0,1), x \sim p_t} \left[ \| u^{\theta}_t(x) - u_t(x) \|^2 \right] \\
&= \mathbb{E}_{t \sim \mathcal{U}(0,1), z \sim p_{data}, x \sim p_t(\cdot|z)} \left[ \| u^{\theta}_t(x) - u_t(x) \|^2 \right]
\end{align}$$

The calculation will be something like this:
1. draw a random time $t$ from a uniform distribution,
2. sample a data point $z$ from the data distribution,
3. sample a point $x$ from the conditional probability path
$$p_t(\cdot|z)$$,
4. compute the vector field $u^{\theta}_t(x)$ by the neural network,
5. finally, compute the loss between $u^{\theta}_t(x)$ and $u_t(x)$.

Seems straightforward, but the target vector field $u_t(x)$ is intractable. However, we can use the conditional vector field
$$u_t(x|z)$$ to construct an unbiased estimator of the loss:

$$\begin{align}
\mathcal{L}_{CFM}(\theta)
&= \mathbb{E}_{t \sim \mathcal{U}(0,1), z \sim p_{data}, x \sim p_t(\cdot|z)} \left[ \| u^{\theta}_t(x) - u_t(x|z) \|^2 \right]
\end{align}$$

This is valid because the marginal flow matching loss equals the conditional flow matching loss up to a constant:

$$
\mathcal{L}_{FM}(\theta) = \mathcal{L}_{CFM}(\theta) + C
$$

Therefore, flow matching training consists of minimizing the conditional flow matching loss.

### Flow Matching for Gaussian Probability Paths
At last, let us take a look at the specific case of Gaussian probability paths for calculating the flow matching loss.
1. Choose the Gaussian probability path
$$p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 \mathbf{I}_d)$$.
2. Sample $x_t$ from $X_t \sim \alpha_t z + \beta_t \epsilon$, where $\epsilon \sim \mathcal{N}(0, \mathbf{I}_d)$.
3. The conditional vector field of $X_t$ can be derived as:
$$u_t(x|z) = (\alpha_t' - (\beta_t'/\beta_t) \alpha_t)z + (\beta_t' / \beta_t)x$$.
4. Then, plug the above formulas into the conditional flow matching loss:

    $$\begin{align}
    \mathcal{L}_{CFM}(\theta)
    &= \mathbb{E}_{t \sim \mathcal{U}(0,1), z \sim p_{data}, x \sim p_t(\cdot|z)} \left[ \| u^{\theta}_t(x) - (\alpha_t' - (\beta_t'/\beta_t) \alpha_t)z - (\beta_t' / \beta_t)x) \|^2 \right] \\
    &= \mathbb{E}_{t \sim \mathcal{U}(0,1), z \sim p_{data}, x \sim p_t(\cdot|z)} \left[ \| u^{\theta}_t(\alpha_t z + \beta_t \epsilon) - (\alpha_t' z + \beta_t' \epsilon) \|^2 \right] \\
    \end{align}$$

5. Finally, let's make the noise schedule concrete by choosing $\alpha_t = t$ and $\beta_t = 1 - t$, then we have $\alpha_t' = 1$ and $\beta_t' = -1$, so that the loss can be simplified as:

    $$\begin{align}
    \mathcal{L}_{CFM}(\theta)
    &= \mathbb{E}_{t \sim \mathcal{U}(0,1), z \sim p_{data}, x \sim p_t(\cdot|z)} \left[ \| u^{\theta}_t(t z + (1-t) \epsilon) - (z - \epsilon) \|^2 \right] \\
    \end{align}$$

## The Big Question

## Conclusion

## References