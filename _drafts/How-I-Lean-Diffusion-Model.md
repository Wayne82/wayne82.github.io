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
The first time I experienced the "magic" of text to image generation was through the DALL-E from OpenAI and in the Discord group of Midjourney almost 4 years ago. Though it was very impressive, it didn't trigger my immediate interest, until the Stable Diffusion was released by Stability AI, which also open-sourced the code and model weights. And its the mechanism of creating high-quality images from pure Gaussian noise really caught my deep curiosity.

This was where I started, and finally I decided to spend most of my spare time in the past 3 months learning the diffusion model.

## The Learning Plan
After several days of exploring the web, reading blog posts, and consulting with LLMs, I have converged on a learning path structured into five key stages:

1. **Conceptual Foundations**: This [guest youtube video from 3Blue1Brown](https://www.youtube.com/watch?v=iv-5mZ_9CPY) gives a good general introduction to diffusion models, besides there are also a ton of good resources listed in the video description. Furthermore, this MIT open course [Generative AI - Text-to-Image Models](https://www.youtube.com/watch?v=NQBhhRG-Pe4) provides a slight more technical yet still accessible introduction to this topic, but still at an entry level.
2. **Mathematical Explanation**: This is where the real challenge lies. I initially dove into lilianweng's blog post [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) and the original paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Ho et al. However, I quickly realized I am not equipped with the necessary mathematical knowledge to understand the flow of the explanation or the descriptions in the paper. This led me to this comprehensive online course [MIT 6.S184: Flow Matching and Diffusion Models](https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH) by Peter Holderrieth - a systematic and rigorous resource for understanding the formulations behind diffusion and flow-based models.
3. **Model Architecture**: The next goal is to understand the model architecture, specifically the **U-Nets** and **Diffusion Transformers (DiTs)**, about how they are structured and trained.
4. **Hands-on Implementation**: As usual, after understanding the theoretical aspects well enough, I will try to implement a toy diffusion model from scratch.
5. **Evolution and Research**: Finally, I want to trace the development history of this field to fulfill my curiosity of the process of discovery, and keep up with the latest research papers and advancements in diffusion models, to understand how the field is evolving and where it might be headed in the future.

At this point, I am deep in the second stage. Navigating the mathematical theory behind the model is really a challenge for me, as I had to pause frequently to build the knowledge base in Probability Theory, Stochastic Differential Equations, Variational Bayesian methods, etc. It is not to become math expert in these areas, but to reach a level where I can understand the mathematical explanations in the original papers and many of the great blog posts out there. I think I am making some progress, but it is still a long way to go.

## What I Understand So Far
1. The problem in mathematical terms.
2. Find the probability path.
3. ODE, vector field, and flow model.
4. Connection between vector field and the probability path, based on marginalization trick and continuity equation.
5. Learning the marginal vector field by conditional flow matching loss.
6. Choose Gaussian Conditional Probability Paths for actual implementation.

## The Big Question

## Conclusion

## References